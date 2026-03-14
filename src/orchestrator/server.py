"""MCP server (Option B) — LangGraph pipeline with CLI subprocesses.

Loads .env automatically so API keys don't need to be in MCP config.

v0.3 features:
    - Checkpoints: thread_id support for multi-turn chains
    - Time-travel: history() and rewind() tools
    - Self-reflection: critique nodes score output quality
    - Self-correction: architect validates plans against codebase
"""

import uuid

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()  # Load .env before anything reads env vars

from .cli_server_pkg.helpers.prompts import build_prompt
from .cli_server_pkg.session.runners import run_claude, run_gemini
from .config import load_config
from .graph import build_orchestrator_graph
from .prompts import ARCHITECT_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from .router import Router

# Load config (still needed for classify model via API)
config = load_config()
router = Router(config)

# Graph is built lazily on first chain() call
_orchestrator_graph = None


def _get_graph():
    """Get or build the orchestrator graph (lazy singleton)."""
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph(config)
    return _orchestrator_graph


# Create the MCP server
mcp = FastMCP("ai-orchestrator")


@mcp.tool()
async def research(question: str, context: str = "") -> str:
    """Deep research using Gemini CLI. Use for domain exploration, technology
    investigation, or understanding unknowns before planning.

    Args:
        question: What you want to research.
        context: Optional context — file contents, prior findings, etc.
    """
    prompt = build_prompt(
        RESEARCH_SYSTEM_PROMPT,
        question,
        f"## Context\n\n{context}" if context else "",
    )
    return await run_gemini(prompt)


@mcp.tool()
async def architect(goal: str, context: str = "", constraints: str = "") -> str:
    """Design an implementation plan using Claude Code CLI.

    Args:
        goal: What you want to build or change.
        context: Optional context — relevant code, file contents, prior research.
        constraints: Optional constraints — tech stack, patterns to follow, etc.
    """
    prompt = build_prompt(
        ARCHITECT_SYSTEM_PROMPT,
        goal,
        f"## Context\n\n{context}" if context else "",
        f"## Constraints\n\n{constraints}" if constraints else "",
    )
    return await run_claude(prompt)


@mcp.tool()
async def classify(task_description: str) -> str:
    """Classify a task into a tier (research / architect / implement) and
    recommend the right pipeline. Uses a fast, cheap API model.

    Args:
        task_description: Description of the task to classify.
    """
    result = await router.classify(task_description)
    tier = result.get("tier", "unknown")
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")
    pipeline = " → ".join(result.get("pipeline", []))

    return (
        f"**Tier:** {tier} (confidence: {confidence:.0%})\n"
        f"**Pipeline:** {pipeline}\n"
        f"**Reasoning:** {reasoning}"
    )


@mcp.tool()
async def chain(task_description: str, context: str = "", thread_id: str = "") -> str:
    """Auto-route a task through the full LangGraph pipeline with checkpoints.

    A dynamic supervisor inspects the state after every node and decides
    what to do next — research, architect, implement, validate, or finish.
    Each node uses CLI tools with native codebase access.

    Pass a thread_id to continue a previous chain (multi-turn). Omit for a
    new thread. The thread_id is returned in the response for follow-ups.

    Args:
        task_description: What you want to accomplish.
        context: Optional context — relevant code, file contents, etc.
        thread_id: Optional thread ID for multi-turn chains. Omit for new.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph_config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"task": task_description}
    if context:
        initial_state["context"] = context

    graph = _get_graph()
    result = await graph.ainvoke(initial_state, config=graph_config)

    formatted = _format_graph_result(result)
    return f"**Thread:** `{thread_id}`\n\n{formatted}"


@mcp.tool()
async def history(thread_id: str, limit: int = 10) -> str:
    """Show the checkpoint history for a chain thread. Use this to see
    what happened at each step — classification, research, architecture, etc.

    Each checkpoint has an ID you can use with rewind() to go back to
    that point and re-run with different parameters.

    Args:
        thread_id: The thread ID from a previous chain() call.
        limit: Max number of checkpoints to show (default 10).
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    entries = []
    async for snapshot in graph.aget_state_history(config, limit=limit):
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id", "?")
        metadata = snapshot.metadata or {}
        step = metadata.get("step", "?")
        source = metadata.get("source", "?")

        # Summarize what's in the state at this checkpoint
        values = snapshot.values or {}
        has = [k for k in ["research_findings", "architecture_plan",
                           "implementation_result"] if values.get(k)]

        entry = (
            f"### Step {step} (`{checkpoint_id[:12]}...`)\n"
            f"- **Source:** {source}\n"
            f"- **Has:** {', '.join(has) if has else 'empty'}\n"
            f"- **Next:** {', '.join(snapshot.next) if snapshot.next else 'END'}"
        )

        # Show supervisor decision if present
        rationale = values.get("supervisor_rationale", "")
        next_node = values.get("next_node", "")
        if rationale:
            entry += f"\n- **Supervisor → {next_node}:** {rationale}"

        # Show validation score if present
        v_score = values.get("validation_score")
        if v_score is not None:
            entry += f"\n- **Validation score:** {v_score:.2f}"

        entries.append(entry)

    if not entries:
        return f"No history found for thread `{thread_id}`."

    return f"## History for thread `{thread_id}`\n\n" + "\n\n".join(entries)


@mcp.tool()
async def rewind(thread_id: str, checkpoint_id: str, new_task: str = "") -> str:
    """Rewind to a previous checkpoint and re-run the graph from that point.

    Use history() to see available checkpoints. Then pass the checkpoint_id
    to rewind to that state. Optionally provide a new task description to
    change what happens next (e.g., re-run architect with different constraints).

    Args:
        thread_id: The thread ID.
        checkpoint_id: The checkpoint ID to rewind to (from history()).
        new_task: Optional new task description. If empty, continues with the original.
    """
    graph = _get_graph()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    # Get the state at the target checkpoint
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return f"No checkpoint found for `{checkpoint_id}` in thread `{thread_id}`."

    # Prepare input for re-run
    input_state = {}
    if new_task:
        input_state["task"] = new_task

    # Re-invoke from that checkpoint
    result = await graph.ainvoke(input_state or None, config=config)

    formatted = _format_graph_result(result)
    next_nodes = ", ".join(snapshot.next) if snapshot.next else "END"
    return (
        f"## Rewound to checkpoint `{checkpoint_id[:12]}...`\n\n"
        f"**Resumed from:** {next_nodes}\n"
        f"**Thread:** `{thread_id}`\n\n"
        f"{formatted}"
    )


def _format_graph_result(state: dict) -> str:
    """Format the final graph state into a readable markdown response."""
    output_parts: list[str] = []

    # Show supervisor journey
    history = state.get("history", [])
    node_calls = state.get("node_calls", {})
    if history:
        journey = "\n".join(f"{i+1}. {h}" for i, h in enumerate(history))
        calls = ", ".join(f"{k}: {v}" for k, v in sorted(node_calls.items()))
        output_parts.append(
            f"## Supervisor Journey\n\n{journey}\n\n**Node calls:** {calls}"
        )

    # Show validation score if present
    v_score = state.get("validation_score")
    if v_score is not None:
        v_feedback = state.get("validation_feedback", "")
        score_line = f"**Validation score:** {v_score:.2f}"
        if v_feedback:
            score_line += f" — {v_feedback}"
        output_parts.append(f"## Quality\n\n{score_line}")

    research_findings = state.get("research_findings", "")
    if research_findings:
        output_parts.append(f"## Research Findings\n\n{research_findings}")

    architecture_plan = state.get("architecture_plan", "")
    if architecture_plan:
        output_parts.append(f"## Architecture Plan\n\n{architecture_plan}")

    implementation_result = state.get("implementation_result", "")
    if implementation_result:
        output_parts.append(f"## Implementation\n\n{implementation_result}")

    if not output_parts:
        return "No output produced by the orchestrator graph."

    return "\n\n---\n\n".join(output_parts)


def main():
    """Entry point — run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
