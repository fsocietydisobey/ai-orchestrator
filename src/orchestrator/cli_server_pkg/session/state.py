"""Session IDs and usage stats for Claude and Gemini."""

import time

# Active session IDs — enable --resume across calls within one MCP server lifetime.
claude_session_id: str | None = None
gemini_session_id: str | None = None

_session_stats = {
    "start_time": time.time(),
    "claude_calls": 0,
    "gemini_calls": 0,
    "claude_tools": {},
    "gemini_tools": {},
}


def track_call(model: str, tool_name: str) -> None:
    """Track a tool call for usage reporting."""
    key = f"{model}_calls"
    _session_stats[key] = _session_stats.get(key, 0) + 1
    tools_key = f"{model}_tools"
    _session_stats[tools_key][tool_name] = _session_stats[tools_key].get(tool_name, 0) + 1


def get_session_stats():
    """Return the session stats dict (for usage tools)."""
    return _session_stats


def clear_sessions() -> list[str]:
    """Clear both session IDs; return list of cleared session descriptions."""
    global claude_session_id, gemini_session_id
    cleared = []
    if claude_session_id:
        cleared.append(f"Claude (`{claude_session_id}`)")
        claude_session_id = None
    if gemini_session_id:
        cleared.append(f"Gemini (`{gemini_session_id}`)")
        gemini_session_id = None
    return cleared


def reset_sessions() -> None:
    """Reset both session IDs without returning info (e.g. for orchestrate pipeline)."""
    global claude_session_id, gemini_session_id
    claude_session_id = None
    gemini_session_id = None
