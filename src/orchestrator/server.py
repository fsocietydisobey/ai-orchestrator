"""MCP server — exposes research, architect, classify, and chain tools.

Direct tools (research, architect, classify) use the existing Router + raw
providers for fast, cheap calls. The chain() tool uses the LangGraph graph
internally, giving agents filesystem access and automatic state flow.
"""

from mcp.server.fastmcp import FastMCP

from .config import load_config
from .graph import build_orchestrator_graph
from .prompts import ARCHITECT_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from .router import Router

# Load config and create router (for direct tools)
config = load_config()
router = Router(config)

# Graph is built lazily on first chain() call to avoid requiring API keys
# at import time (MCP server may start before keys are configured)
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
    """Deep research using Gemini. Use for domain exploration, technology
    investigation, or understanding unknowns before planning.

    Args:
        question: What you want to research.
        context: Optional context — file contents, prior findings, etc.
    """
    prompt = question
    if context:
        prompt = f"{question}\n\n## Context\n\n{context}"

    result = await router.route(
        "research", prompt, system_prompt=RESEARCH_SYSTEM_PROMPT
    )
    provider = router.get_role_provider("research")
    return f"**[{provider.name}]**\n\n{result}"


@mcp.tool()
async def architect(goal: str, context: str = "", constraints: str = "") -> str:
    """Design an implementation plan using Claude. Use when the problem is
    understood but the solution needs design — multi-file changes, API design,
    architecture decisions.

    Args:
        goal: What you want to build or change.
        context: Optional context — relevant code, file contents, prior research.
        constraints: Optional constraints — tech stack, patterns to follow, etc.
    """
    parts = [goal]
    if context:
        parts.append(f"## Context\n\n{context}")
    if constraints:
        parts.append(f"## Constraints\n\n{constraints}")
    prompt = "\n\n".join(parts)

    result = await router.route(
        "architect", prompt, system_prompt=ARCHITECT_SYSTEM_PROMPT
    )
    provider = router.get_role_provider("architect")
    return f"**[{provider.name}]**\n\n{result}"


@mcp.tool()
async def classify(task_description: str) -> str:
    """Classify a task into a tier (research / architect / implement) and
    recommend the right pipeline. Uses a fast, cheap model.

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
async def chain(task_description: str, context: str = "") -> str:
    """Auto-route a task through the full pipeline using LangGraph. Classifies
    first, then runs research (if needed) and architecture (if needed).

    Unlike the direct tools, chain() gives each model access to filesystem
    tools — they can read files, grep code, and explore directories
    autonomously. State flows between nodes automatically.

    This is the "do the thinking for me" tool — it figures out what kind of
    task this is and runs the appropriate agents in sequence.

    Args:
        task_description: What you want to accomplish.
        context: Optional context — relevant code, file contents, etc.
    """
    # Build initial state for the graph
    initial_state = {"task": task_description}
    if context:
        initial_state["context"] = context

    # Invoke the LangGraph orchestrator (builds graph on first call)
    graph = _get_graph()
    result = await graph.ainvoke(initial_state)

    return _format_graph_result(result)


def _format_graph_result(state: dict) -> str:
    """Format the final graph state into a readable markdown response.

    Assembles output sections from whichever fields the graph populated:
    classification, research findings, architecture plan, implementation.
    """
    output_parts: list[str] = []

    # Classification section
    classification = state.get("classification", {})
    if classification:
        tier = classification.get("tier", "unknown")
        pipeline = classification.get("pipeline", [])
        reasoning = classification.get("reasoning", "")
        output_parts.append(
            f"## Classification\n\n"
            f"**Tier:** {tier}\n"
            f"**Pipeline:** {' → '.join(pipeline)}\n"
            f"**Reasoning:** {reasoning}"
        )

    # Research findings section
    research_findings = state.get("research_findings", "")
    if research_findings:
        output_parts.append(f"## Research Findings\n\n{research_findings}")

    # Architecture plan section
    architecture_plan = state.get("architecture_plan", "")
    if architecture_plan:
        output_parts.append(f"## Implementation Plan\n\n{architecture_plan}")

    # Implementation result section (placeholder for now)
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
