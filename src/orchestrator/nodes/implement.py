"""Implement node — shells out to Claude Code CLI for code generation.

Claude Code runs from the project root with full read/write codebase access.
Takes the architecture plan and supervisor instructions to implement changes.
"""

from ..cli_server_pkg.helpers.prompts import build_prompt
from ..cli_server_pkg.session.runners import run_claude
from ..state import OrchestratorState

IMPLEMENT_SYSTEM_PROMPT = """\
You are a senior software engineer. Your job is to implement code changes
based on an architecture plan. You have full codebase access.

## How you work

1. Read the architecture plan carefully.
2. Read the relevant source files to understand existing code.
3. Implement each step in order — make the actual file changes.
4. Verify your changes are consistent with existing patterns and conventions.

## Rules

- Follow the plan precisely. Don't redesign — implement.
- Match existing code style, naming conventions, and patterns.
- If a step is unclear, make a reasonable choice and note it.
- Do NOT skip steps or leave TODOs for later.
"""


def build_implement_node():
    """Build an implement node that uses Claude Code CLI.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """

    async def implement_node(state: OrchestratorState) -> dict:
        """Run Claude Code implementation based on the architecture plan."""
        task = state.get("task", "")
        context = state.get("context", "")
        architecture_plan = state.get("architecture_plan", "")
        instructions = state.get("supervisor_instructions", "")
        node_calls = dict(state.get("node_calls", {}))
        history = list(state.get("history", []))

        # Track call count
        node_calls["implement"] = node_calls.get("implement", 0) + 1

        prompt = build_prompt(
            IMPLEMENT_SYSTEM_PROMPT,
            f"## Task\n\n{task}",
            f"## Context\n\n{context}" if context else "",
            f"## Architecture Plan\n\n{architecture_plan}" if architecture_plan else "",
            f"## Additional instructions\n\n{instructions}" if instructions else "",
        )

        result = await run_claude(prompt, timeout=600)

        return {
            "implementation_result": result,
            "node_calls": node_calls,
            "history": history + [f"implement: completed (attempt {node_calls['implement']})"],
        }

    return implement_node
