"""Gemini MCP tools: research, explain, compare."""

from ..helpers.prompts import build_prompt
from ..session.runners import run_gemini
from ..session.state import track_call


def register_gemini_tools(mcp):
    """Register Gemini tools on the given FastMCP instance."""

    @mcp.tool()
    async def research(question: str, context: str = "") -> str:
        """Deep research using Gemini. Use for domain exploration, technology
        investigation, or understanding unknowns before planning.

        Gemini has strong research and analysis skills. Do NOT use this for
        writing code — use it for understanding problems, exploring options,
        and gathering information.

        Args:
            question: What you want to research.
            context: Optional extra context to include in the prompt.
        """
        prompt = build_prompt(
            question,
            f"## Context\n\n{context}" if context else "",
        )
        track_call("gemini", "research")
        return await run_gemini(prompt)

    @mcp.tool()
    async def explain(code_or_concept: str, context: str = "") -> str:
        """Explain unfamiliar code, concepts, or patterns using Gemini.

        Use when you encounter something you don't understand — a complex
        function, an unfamiliar library pattern, a design pattern, or a
        concept. Gemini will research and explain it clearly.

        Args:
            code_or_concept: The code snippet, function name, or concept to explain.
            context: Optional context — where this code lives, what you're trying to do.
        """
        prompt = build_prompt(
            "Explain the following clearly and thoroughly. Break down how it works, "
            "why it's done this way, and call out any non-obvious behavior or gotchas.\n",
            code_or_concept,
            f"## Context\n\n{context}" if context else "",
        )
        track_call("gemini", "explain")
        return await run_gemini(prompt)

    @mcp.tool()
    async def compare(option_a: str, option_b: str, context: str = "") -> str:
        """Compare two approaches, technologies, or options using Gemini.

        Use when deciding between alternatives — libraries, design patterns,
        architectures, or implementation strategies. Gemini will research
        both and present trade-offs.

        Args:
            option_a: First option or approach.
            option_b: Second option or approach.
            context: Optional context — what you're building, constraints, priorities.
        """
        prompt = build_prompt(
            "Compare the following two options. For each, cover: strengths, weaknesses, "
            "trade-offs, and when to prefer one over the other. End with a recommendation.\n",
            f"## Option A\n\n{option_a}",
            f"## Option B\n\n{option_b}",
            f"## Context\n\n{context}" if context else "",
        )
        track_call("gemini", "compare")
        return await run_gemini(prompt)
