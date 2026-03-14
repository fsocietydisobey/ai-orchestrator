"""Architect node — shells out to Claude Code CLI for design/planning.

Claude Code runs from the project root with native codebase access.
Includes self-correction: validates file paths and function names against
the actual codebase before finalizing.
"""

from ..cli_server_pkg.helpers.prompts import build_prompt
from ..cli_server_pkg.session.runners import run_claude
from ..prompts import ARCHITECT_SYSTEM_PROMPT
from ..state import OrchestratorState


def build_architect_node():
    """Build an architect node that uses Claude Code CLI.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """

    async def architect_node(state: OrchestratorState) -> dict:
        """Run Claude Code architecture and return the plan."""
        task = state.get("task", "")
        context = state.get("context", "")
        research_findings = state.get("research_findings", "")
        instructions = state.get("supervisor_instructions", "")
        feedback = state.get("validation_feedback", "")
        node_calls = dict(state.get("node_calls", {}))
        history = list(state.get("history", []))

        # Track call count
        node_calls["architect"] = node_calls.get("architect", 0) + 1

        prompt = build_prompt(
            ARCHITECT_SYSTEM_PROMPT,
            task,
            f"## Context\n\n{context}" if context else "",
            f"## Research Findings\n\n{research_findings}" if research_findings else "",
            f"## Supervisor instructions\n\n{instructions}" if instructions else "",
            f"## Previous feedback to address\n\n{feedback}" if feedback else "",
            # Self-correction
            "## Self-correction\n\n"
            "Before finalizing your plan, verify that every file path and function "
            "name you reference actually exists in the codebase. Read the files to "
            "confirm. If you find a hallucinated path or name, fix it.",
        )

        plan = await run_claude(prompt, timeout=600)

        return {
            "architecture_plan": plan,
            "node_calls": node_calls,
            "history": history + [f"architect: completed (attempt {node_calls['architect']})"],
        }

    return architect_node
