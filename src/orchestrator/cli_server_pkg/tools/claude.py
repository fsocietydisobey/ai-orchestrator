"""Claude MCP tools: architect, implement, review, debug, test, document."""

from ..helpers.prompts import build_prompt
from ..session.runners import run_claude
from ..session.state import track_call


def register_claude_tools(mcp):
    """Register Claude tools on the given FastMCP instance."""

    @mcp.tool()
    async def architect(goal: str, context: str = "", constraints: str = "") -> str:
        """Design an implementation plan using Claude Code. Use for complex
        architecture decisions, multi-file coordination, and tasks that need
        deep codebase understanding.

        Claude Code runs with full codebase access — it can read files, search
        code, and understand project structure natively.

        Args:
            goal: What you want to design or plan.
            context: Optional context — relevant background, prior research.
            constraints: Optional constraints — tech stack, patterns to follow.
        """
        prompt = build_prompt(
            "You are a senior software architect. Design a detailed implementation plan "
            "with specific file paths, function names, and step-by-step changes.\n",
            goal,
            f"## Context\n\n{context}" if context else "",
            f"## Constraints\n\n{constraints}" if constraints else "",
        )
        track_call("claude", "architect")
        return await run_claude(prompt)

    @mcp.tool()
    async def implement(spec: str, context: str = "") -> str:
        """Implement a complex task using Claude Code. Use this for tasks that
        are too complex for Cursor's built-in models — intricate refactors,
        multi-file changes, or tricky logic.

        Claude Code runs with full codebase access and can read/write files
        directly. It will analyze the codebase and produce the implementation.

        Args:
            spec: What to implement — be specific about files, functions, behavior.
            context: Optional context — architecture plan, research findings, etc.
        """
        prompt = build_prompt(
            spec,
            f"## Context\n\n{context}" if context else "",
        )
        track_call("claude", "implement")
        return await run_claude(prompt)

    @mcp.tool()
    async def review(target: str, focus: str = "") -> str:
        """Thorough code review using Claude Code. Use before PRs or when you
        want a second opinion on code quality, correctness, or security.

        Claude Code will read the relevant files and provide a detailed review
        covering bugs, security issues, performance, readability, and suggestions.

        Args:
            target: What to review — a file path, diff description, or code snippet.
            focus: Optional focus area — 'security', 'performance', 'correctness', etc.
        """
        prompt = build_prompt(
            "You are a senior code reviewer. Review the following thoroughly. "
            "Check for: bugs, security vulnerabilities, performance issues, "
            "readability problems, and missed edge cases. Be specific — reference "
            "line numbers and suggest concrete fixes.\n",
            target,
            f"## Focus area\n\n{focus}" if focus else "",
        )
        track_call("claude", "review")
        return await run_claude(prompt)

    @mcp.tool()
    async def debug(error: str, context: str = "") -> str:
        """Root cause analysis using Claude Code. Use when you have an error,
        stack trace, or unexpected behavior and need to figure out why.

        Claude Code will read the relevant source files, trace the execution
        path, and identify the root cause with a fix.

        Args:
            error: The error message, stack trace, or description of unexpected behavior.
            context: Optional context — what you were doing, recent changes, etc.
        """
        prompt = build_prompt(
            "You are debugging an issue. Analyze the error below, read the relevant "
            "source files, trace the execution path, and identify the root cause. "
            "Provide a clear explanation and a concrete fix.\n",
            f"## Error\n\n{error}",
            f"## Context\n\n{context}" if context else "",
        )
        track_call("claude", "debug")
        return await run_claude(prompt)

    @mcp.tool()
    async def test(target: str, context: str = "") -> str:
        """Generate test cases using Claude Code. Use when you need tests for
        a function, module, or feature.

        Claude Code will read the source code, understand the behavior, and
        generate comprehensive test cases covering happy paths, edge cases,
        and error scenarios.

        Args:
            target: What to test — a function name, file path, or feature description.
            context: Optional context — test framework to use, specific scenarios to cover.
        """
        prompt = build_prompt(
            "Generate comprehensive test cases for the following. Cover: happy paths, "
            "edge cases, error scenarios, and boundary conditions. Use the project's "
            "existing test framework and patterns if any exist.\n",
            target,
            f"## Context\n\n{context}" if context else "",
        )
        track_call("claude", "test")
        return await run_claude(prompt)

    @mcp.tool()
    async def document(target: str, style: str = "") -> str:
        """Generate documentation using Claude Code. Use when you need docs
        for a module, API, function, or feature.

        Claude Code will read the source code and generate clear, accurate
        documentation.

        Args:
            target: What to document — a file path, module name, or feature description.
            style: Optional style — 'docstrings', 'readme', 'api reference', 'tutorial', etc.
        """
        prompt = build_prompt(
            "Generate clear, accurate documentation for the following. Read the source "
            "code to understand the actual behavior — don't guess or hallucinate.\n",
            target,
            f"## Style\n\n{style}" if style else "",
        )
        track_call("claude", "document")
        return await run_claude(prompt)
