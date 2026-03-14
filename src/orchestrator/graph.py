"""LangGraph StateGraph — v0.4 Dynamic Supervisor pattern.

Hub-and-spoke architecture: every node returns to the supervisor, which
inspects the full state and decides what to do next (or terminate).

    supervisor → research → supervisor → architect → supervisor → implement → supervisor → END
             ↑                                                                    |
             └──────────────── every node returns here ───────────────────────────┘

Features:
    - Dynamic routing via Pydantic RouterDecision (structured LLM output)
    - Self-healing: supervisor detects failure and retries with feedback
    - Checkpoints via InMemorySaver for time-travel
    - Supervisor uses cheap API model (Haiku). Domain nodes use CLI subprocesses.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .config import OrchestratorConfig
from .models import get_classify_model
from .nodes import (
    build_architect_node,
    build_implement_node,
    build_research_node,
    build_supervisor_node,
    build_validator_node,
)
from .state import OrchestratorState

# Shared checkpointer — persists state across invocations within the server lifetime
checkpointer = InMemorySaver()


def select_next_node(state: OrchestratorState) -> str:
    """Read the supervisor's decision and route to the next node."""
    next_node = state.get("next_node", "finish")
    if next_node == "finish":
        return END
    return next_node


def build_orchestrator_graph(config: OrchestratorConfig):
    """Build and compile the orchestrator StateGraph with dynamic supervisor.

    The supervisor (Haiku) makes routing decisions. Domain nodes (research,
    architect, implement) use CLI subprocesses. Validator (Haiku) scores output.

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

    # --- Wire the graph ---
    graph = StateGraph(OrchestratorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("architect", architect_node)
    graph.add_node("implement", implement_node)
    graph.add_node("validator", validator_node)

    # Entry: START → supervisor (supervisor makes the first decision)
    graph.add_edge(START, "supervisor")

    # Supervisor routes dynamically
    graph.add_conditional_edges(
        "supervisor",
        select_next_node,
        {
            "research": "research",
            "architect": "architect",
            "implement": "implement",
            "validator": "validator",
            END: END,
        },
    )

    # Every node returns to supervisor
    graph.add_edge("research", "supervisor")
    graph.add_edge("architect", "supervisor")
    graph.add_edge("implement", "supervisor")
    graph.add_edge("validator", "supervisor")

    return graph.compile(checkpointer=checkpointer)
