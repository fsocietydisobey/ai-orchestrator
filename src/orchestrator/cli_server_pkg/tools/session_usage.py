"""Session and usage MCP tools: new_session, claude_usage, gemini_usage."""

import json
import time
from pathlib import Path

from .. import config
from ..session.state import clear_sessions, get_session_stats


def register_session_usage_tools(mcp):
    """Register session and usage tools on the given FastMCP instance."""

    @mcp.tool()
    async def new_session() -> str:
        """Reset both Claude and Gemini sessions. Next call to either model
        starts a fresh conversation with no prior context.

        Use when switching to a completely different task and you don't want
        prior context bleeding in.
        """
        cleared = clear_sessions()
        if cleared:
            return f"Sessions cleared: {', '.join(cleared)}. Next calls start fresh."
        return "No active sessions. Next calls start fresh."

    @mcp.tool()
    async def claude_usage() -> str:
        """Show Claude Code usage — token counts and costs from recent sessions.

        Reads Claude Code's session logs to show how many tokens were used
        and approximate costs. Shows both the current MCP server session stats
        and historical Claude Code session data.
        """
        _session_stats = get_session_stats()
        parts: list[str] = []

        uptime_s = time.time() - _session_stats["start_time"]
        uptime_m = int(uptime_s // 60)
        parts.append(
            f"## MCP Server Session\n\n"
            f"**Uptime:** {uptime_m}m\n"
            f"**Claude calls this session:** {_session_stats['claude_calls']}"
        )
        if _session_stats["claude_tools"]:
            breakdown = ", ".join(
                f"{name}: {count}" for name, count in sorted(_session_stats["claude_tools"].items())
            )
            parts.append(f"**Breakdown:** {breakdown}")

        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            parts.append("\n## Historical Usage\n\nNo Claude Code session data found.")
            return "\n".join(parts)

        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_create = 0
        session_count = 0
        cutoff = time.time() - (7 * 86400)

        for project_dir in claude_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for session_file in project_dir.glob("*.jsonl"):
                if session_file.stat().st_mtime < cutoff:
                    continue
                session_count += 1
                try:
                    for line in session_file.read_text().splitlines():
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = entry.get("message", {})
                        usage = msg.get("usage", {})
                        if usage:
                            total_input += usage.get("input_tokens", 0)
                            total_output += usage.get("output_tokens", 0)
                            total_cache_read += usage.get("cache_read_input_tokens", 0)
                            total_cache_create += usage.get("cache_creation_input_tokens", 0)
                except Exception:
                    continue

        input_cost = (total_input / 1_000_000) * 15.0
        output_cost = (total_output / 1_000_000) * 75.0
        cache_read_cost = (total_cache_read / 1_000_000) * 1.875
        cache_create_cost = (total_cache_create / 1_000_000) * 18.75
        total_cost = input_cost + output_cost + cache_read_cost + cache_create_cost

        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        parts.append(
            f"\n## Claude Code Usage (last 7 days)\n\n"
            f"**Sessions:** {session_count}\n"
            f"**Input tokens:** {_fmt(total_input)}\n"
            f"**Output tokens:** {_fmt(total_output)}\n"
            f"**Cache read tokens:** {_fmt(total_cache_read)}\n"
            f"**Cache creation tokens:** {_fmt(total_cache_create)}\n"
            f"**Estimated cost:** ${total_cost:.2f}\n\n"
            f"*Cost estimate based on Opus pricing ($15/M input, $75/M output, "
            f"$1.875/M cache read, $18.75/M cache write). Actual cost depends "
            f"on which model was used per session.*"
        )

        return "\n".join(parts)

    @mcp.tool()
    async def gemini_usage() -> str:
        """Show Gemini usage for this MCP server session.

        Gemini CLI doesn't store local usage logs, so this shows call counts
        and breakdown by tool for the current server session only.
        """
        _session_stats = get_session_stats()
        uptime_s = time.time() - _session_stats["start_time"]
        uptime_m = int(uptime_s // 60)

        parts = [
            f"## Gemini Usage (this session)\n\n"
            f"**Uptime:** {uptime_m}m\n"
            f"**Model:** {config.GEMINI_MODEL}\n"
            f"**Total calls:** {_session_stats['gemini_calls']}"
        ]

        if _session_stats["gemini_tools"]:
            breakdown = ", ".join(
                f"{name}: {count}" for name, count in sorted(_session_stats["gemini_tools"].items())
            )
            parts.append(f"**Breakdown:** {breakdown}")
        else:
            parts.append("**Breakdown:** No calls yet")

        parts.append(
            "\n*Gemini CLI doesn't store local usage/cost data. "
            "Check the Google AI Studio console for detailed token usage and billing.*"
        )

        return "\n".join(parts)
