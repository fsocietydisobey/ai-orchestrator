"""LangGraph StateGraph — v0.5 Dynamic Supervisor with Fan-Out and HITL.

Hub-and-spoke architecture: every node returns to the supervisor, which
inspects the full state and decides what to do next (or terminate).

v0.5 features:
    - Fan-out/fan-in: supervisor dispatches parallel research via Send()
    - Human-in-the-loop: graph pauses for human approval before implementation
    - Dynamic routing via Pydantic RouterDecision (structured LLM output)
    - Self-healing: supervisor detects failure and retries with feedback
    - Checkpoints via InMemorySaver for time-travel
    - Supervisor uses cheap API model (Haiku). Domain nodes use CLI subprocesses.

Flow with HITL:
    supervisor → architect → supervisor → human_review (PAUSED)
    → [human approves] → supervisor → implement → supervisor → END
    → [human rejects]  → supervisor → architect (with feedback) → ...
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import OrchestratorConfig
from .models import get_classify_model
from .nodes import (
    build_architect_node,
    build_human_review_node,
    build_implement_node,
    build_research_node,
    build_supervisor_node,
    build_validator_node,
)
from .state import OrchestratorState

# Shared checkpointer — persists state across invocations within the server lifetime
checkpointer = InMemorySaver()


def select_next_node(state: OrchestratorState) -> str | list[Send]:
    """Read the supervisor's decision and route to the next node.

    If parallel_tasks is populated and next_node is 'research',
    fan out via Send(). Otherwise, route sequentially.
    """
    next_node = state.get("next_node", "finish")
    if next_node == "finish":
        return END

    parallel_tasks = state.get("parallel_tasks", [])

    # Fan-out: supervisor chose research AND provided parallel sub-tasks
    if next_node == "research" and parallel_tasks:
        return [
            Send(
                "research",
                {
                    "task": state.get("task", ""),
                    "context": state.get("context", ""),
                    "supervisor_instructions": pt.get("instructions", ""),
                    "parallel_task_topic": pt.get("topic", ""),
                    "validation_feedback": state.get("validation_feedback", ""),
                },
            )
            for pt in parallel_tasks
        ]

    return next_node


def _research_exit(state: OrchestratorState) -> str:
    """Route research output: to merge_research if fan-out, else to supervisor."""
    if state.get("parallel_tasks", []):
        return "merge_research"
    return "supervisor"


async def _merge_research_node(state: OrchestratorState) -> dict:
    """Combine parallel research outputs into a single research_findings string.

    Reads output_versions to find entries from the current parallel batch,
    synthesizes them into sectioned markdown, then clears parallel_tasks.
    """
    output_versions = state.get("output_versions", [])
    parallel_tasks = state.get("parallel_tasks", [])
    history = list(state.get("history", []))

    # Collect research entries matching the current parallel topics
    parallel_topics = {pt.get("topic", "") for pt in parallel_tasks}
    parallel_results = [
        entry
        for entry in output_versions
        if entry.get("node") == "research" and entry.get("topic", "") in parallel_topics
    ]

    # Synthesize into sectioned markdown
    if parallel_results:
        sections = []
        for result in parallel_results:
            topic = result.get("topic", "unknown")
            content = result.get("content", "")
            sections.append(f"### {topic}\n\n{content}")
        merged = "\n\n---\n\n".join(sections)
    else:
        # Fallback: use whatever research_findings exists
        merged = state.get("research_findings", "")

    history.append(f"merge_research: combined {len(parallel_results)} parallel findings")

    return {
        "research_findings": merged,
        "parallel_tasks": [],  # Clear to prevent re-triggering fan-out
        "history": history,
    }


def build_orchestrator_graph(config: OrchestratorConfig):
    """Build and compile the orchestrator StateGraph with dynamic supervisor.

    The supervisor (Haiku) makes routing decisions. Domain nodes (research,
    architect, implement) use CLI subprocesses. Validator (Haiku) scores output.
    Fan-out is supported for research via Send(). Human review pauses
    the graph before implementation.

    Args:
        config: OrchestratorConfig with provider/role definitions.

    Returns:
        Compiled StateGraph with InMemorySaver checkpointer.
    """
    # Supervisor and validator use a cheap/fast API model
    supervisor_model = get_classify_model(config)

    supervisor_node = build_supervisor_node(supervisor_model)
    validator_node = build_validator_node(supervisor_model)
    research_node = build_research_node()
    architect_node = build_architect_node()
    implement_node = build_implement_node()
    human_review_node = build_human_review_node()

    # --- Wire the graph ---
    graph = StateGraph(OrchestratorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("architect", architect_node)
    graph.add_node("implement", implement_node)
    graph.add_node("validator", validator_node)
    graph.add_node("merge_research", _merge_research_node)
    graph.add_node("human_review", human_review_node)

    # Entry: START → supervisor (supervisor makes the first decision)
    graph.add_edge(START, "supervisor")

    # Supervisor routes dynamically (may return Send() list for fan-out)
    graph.add_conditional_edges(
        "supervisor",
        select_next_node,
        {
            "research": "research",
            "architect": "architect",
            "human_review": "human_review",
            "implement": "implement",
            "validator": "validator",
            END: END,
        },
    )

    # Research → conditional: fan-out goes to merge, sequential goes to supervisor
    graph.add_conditional_edges(
        "research",
        _research_exit,
        {
            "merge_research": "merge_research",
            "supervisor": "supervisor",
        },
    )

    # merge_research always returns to supervisor
    graph.add_edge("merge_research", "supervisor")

    # human_review returns to supervisor (which reads approval/rejection)
    graph.add_edge("human_review", "supervisor")

    # Other nodes return to supervisor
    graph.add_edge("architect", "supervisor")
    graph.add_edge("implement", "supervisor")
    graph.add_edge("validator", "supervisor")

    return graph.compile(checkpointer=checkpointer)
