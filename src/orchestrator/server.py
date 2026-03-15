"""MCP server (Option B) — LangGraph pipeline with CLI subprocesses.

Loads .env automatically so API keys don't need to be in MCP config.

v0.5 features:
    - Checkpoints: thread_id support for multi-turn chains
    - Time-travel: history() and rewind() tools
    - Self-correction: validator scores output, supervisor retries with feedback
    - Fan-out: parallel research via Send()
    - HITL: graph pauses for human approval before implementation
"""

import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from mcp.server.fastmcp import Context, FastMCP

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
async def chain(task_description: str, ctx: Context, context: str = "", thread_id: str = "") -> str:
    """Auto-route a task through the full LangGraph pipeline with checkpoints.

    A dynamic supervisor inspects the state after every node and decides
    what to do next — research, architect, implement, validate, or finish.
    Each node uses CLI tools with native codebase access.

    Streams real-time progress updates as each node completes — you can
    watch the supervisor's decisions and node results as they happen.

    The pipeline will PAUSE for human approval before implementation.
    When paused, use approve(thread_id) to continue or approve(thread_id,
    feedback="...") to reject and send the architect back to revise.

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

    # Stream updates instead of invoking — get real-time progress
    result = {}
    step = 0
    async for update in graph.astream(
        initial_state, config=graph_config, stream_mode="updates"
    ):
        step += 1
        # update is {node_name: state_update_dict}
        for node_name, state_update in update.items():
            result.update(state_update)

            # Build a progress message from the supervisor's decision
            message = _build_progress_message(node_name, state_update)
            await ctx.report_progress(step, message=message)

    # Check if graph is paused at a human review interrupt
    state = await graph.aget_state(graph_config)
    if state and state.next and "human_review" in state.next:
        # Graph is paused — extract the review payload from the interrupt
        plan = result.get("architecture_plan", "")
        task = result.get("task", task_description)
        return (
            f"**Thread:** `{thread_id}`\n\n"
            f"## Waiting for Human Approval\n\n"
            f"The architecture plan is ready for review. "
            f"The pipeline is **paused** and will not implement until you approve.\n\n"
            f"### Task\n\n{task}\n\n"
            f"### Architecture Plan\n\n{plan}\n\n"
            f"---\n\n"
            f"**To approve:** `approve(thread_id=\"{thread_id}\")`\n\n"
            f"**To reject with feedback:** `approve(thread_id=\"{thread_id}\", feedback=\"your feedback here\")`"
        )

    formatted = _format_graph_result(result)
    return f"**Thread:** `{thread_id}`\n\n{formatted}"


@mcp.tool()
async def approve(thread_id: str, ctx: Context, feedback: str = "") -> str:
    """Approve or reject a paused chain that is waiting for human review.

    After chain() pauses for human approval, call this to continue.
    Without feedback, the plan is approved and implementation proceeds.
    With feedback, the plan is rejected and the architect revises it.

    Streams real-time progress updates as the pipeline resumes.

    Args:
        thread_id: The thread ID from the paused chain() call.
        feedback: Optional feedback. If provided, the plan is rejected
            and the architect will revise based on your feedback.
    """
    graph = _get_graph()
    graph_config = {"configurable": {"thread_id": thread_id}}

    # Check the graph is actually paused
    state = await graph.aget_state(graph_config)
    if not state or not state.next or "human_review" not in state.next:
        return f"Thread `{thread_id}` is not waiting for approval."

    # Build the resume command
    if feedback:
        resume_value = {"decision": "rejected", "feedback": feedback}
    else:
        resume_value = {"decision": "approved", "feedback": ""}

    # Resume the graph with streaming — Command(resume=...) provides the value to interrupt()
    result = {}
    step = 0
    async for update in graph.astream(
        Command(resume=resume_value),
        config=graph_config,
        stream_mode="updates",
    ):
        step += 1
        for node_name, state_update in update.items():
            result.update(state_update)
            message = _build_progress_message(node_name, state_update)
            await ctx.report_progress(step, message=message)

    # Check if it paused again (e.g., architect revised, hit another review)
    state = await graph.aget_state(graph_config)
    if state and state.next and "human_review" in state.next:
        plan = result.get("architecture_plan", "")
        return (
            f"**Thread:** `{thread_id}`\n\n"
            f"## Revised Plan — Waiting for Approval Again\n\n"
            f"The architect revised the plan based on your feedback. "
            f"Review and approve/reject again.\n\n"
            f"### Architecture Plan\n\n{plan}\n\n"
            f"---\n\n"
            f"**To approve:** `approve(thread_id=\"{thread_id}\")`\n\n"
            f"**To reject with feedback:** `approve(thread_id=\"{thread_id}\", feedback=\"...\")`"
        )

    formatted = _format_graph_result(result)
    decision = "rejected" if feedback else "approved"
    return (
        f"**Thread:** `{thread_id}`\n\n"
        f"## Review: {decision}\n\n"
        f"{formatted}"
    )


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

        # Show human review status if present
        review_status = values.get("human_review_status", "")
        if review_status:
            entry += f"\n- **Human review:** {review_status}"

        # Show output version count for timeline comparison
        versions = values.get("output_versions", [])
        if versions:
            version_summary = {}
            for v in versions:
                node = v.get("node", "?")
                version_summary[node] = version_summary.get(node, 0) + 1
            counts = ", ".join(f"{k}: {v}" for k, v in sorted(version_summary.items()))
            entry += f"\n- **Output versions:** {counts}"

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


def _build_progress_message(node_name: str, state_update: dict) -> str:
    """Build a human-readable progress message from a node's state update.

    Called after each node completes during astream(). Returns a short
    message suitable for MCP progress notifications.
    """
    if node_name == "supervisor":
        next_node = state_update.get("next_node", "?")
        rationale = state_update.get("supervisor_rationale", "")
        parallel = state_update.get("parallel_tasks", [])
        msg = f"Supervisor → {next_node}"
        if rationale:
            msg += f": {rationale}"
        if parallel:
            topics = ", ".join(pt.get("topic", "?") for pt in parallel)
            msg += f" [fan-out: {topics}]"
        return msg

    if node_name == "validator":
        score = state_update.get("validation_score")
        feedback = state_update.get("validation_feedback", "")
        if score is not None:
            msg = f"Validator: score {score:.2f}"
            if feedback:
                msg += f" — {feedback}"
            return msg
        return "Validator: scoring output"

    if node_name == "research":
        topic = state_update.get("parallel_task_topic", "")
        if topic:
            return f"Research completed: {topic}"
        return "Research completed"

    if node_name == "architect":
        return "Architect: plan ready"

    if node_name == "implement":
        return "Implementation completed"

    if node_name == "merge_research":
        return "Merge: combining parallel research findings"

    if node_name == "human_review":
        status = state_update.get("human_review_status", "")
        if status:
            return f"Human review: {status}"
        return "Human review: waiting for approval"

    return f"{node_name}: completed"


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

    # Show human review status if present
    review_status = state.get("human_review_status", "")
    if review_status:
        human_feedback = state.get("human_feedback", "")
        review_line = f"**Human review:** {review_status}"
        if human_feedback:
            review_line += f" — {human_feedback}"
        output_parts.append(f"## Human Review\n\n{review_line}")

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
