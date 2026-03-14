"""Research node — shells out to Gemini CLI for deep exploration.

Gemini CLI runs from the project root with native codebase access.
Incorporates supervisor instructions and validation feedback on retries.
"""

from ..cli_server_pkg.helpers.prompts import build_prompt
from ..cli_server_pkg.session.runners import run_gemini
from ..prompts import RESEARCH_SYSTEM_PROMPT
from ..state import OrchestratorState


def build_research_node():
    """Build a research node that uses Gemini CLI.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """

    async def research_node(state: OrchestratorState) -> dict:
        """Run Gemini CLI research and return findings."""
        task = state.get("task", "")
        context = state.get("context", "")
        instructions = state.get("supervisor_instructions", "")
        feedback = state.get("validation_feedback", "")
        node_calls = dict(state.get("node_calls", {}))
        history = list(state.get("history", []))

        # Track call count
        node_calls["research"] = node_calls.get("research", 0) + 1

        prompt = build_prompt(
            RESEARCH_SYSTEM_PROMPT,
            task,
            f"## Context\n\n{context}" if context else "",
            f"## Supervisor instructions\n\n{instructions}" if instructions else "",
            f"## Previous feedback to address\n\n{feedback}" if feedback else "",
        )

        findings = await run_gemini(prompt, timeout=600)

        return {
            "research_findings": findings,
            "node_calls": node_calls,
            "history": history + [f"research: completed (attempt {node_calls['research']})"],
        }

    return research_node
