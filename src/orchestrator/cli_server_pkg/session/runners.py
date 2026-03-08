"""Run Gemini and Claude CLI subprocesses with session continuity."""

import json

from .. import config
from ..helpers.cli import cli_available, run_cli
from . import state


async def run_gemini(prompt: str, timeout: int | None = None) -> str:
    """Run a prompt through Gemini CLI via npx, continuing the session if one exists."""
    if not cli_available(config.NPX_CMD):
        return f"Error: npx not found at `{config.NPX_CMD}`. Set NPX_CMD env var or check your nvm install."

    cmd = [config.NPX_CMD, "-y", config.GEMINI_PKG, "-m", config.GEMINI_MODEL, "-p", prompt, "-o", "json"]
    if state.gemini_session_id:
        cmd.extend(["--resume", state.gemini_session_id])

    try:
        raw = await run_cli(cmd, timeout=timeout)
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"

    try:
        data = json.loads(raw)
        session_id = data.get("session_id") or data.get("sessionId")
        if session_id:
            state.gemini_session_id = session_id

        result = data.get("result", "") or data.get("response", "") or data.get("text", "")
        if not result:
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
        return result or raw
    except (json.JSONDecodeError, TypeError):
        return raw


async def run_claude(prompt: str, timeout: int | None = None) -> str:
    """Run a prompt through Claude Code CLI, continuing the session if one exists."""
    if not cli_available(config.CLAUDE_CMD):
        return f"Error: Claude CLI not found at `{config.CLAUDE_CMD}`. Set CLAUDE_CMD env var or install Claude Code."

    cmd = [config.CLAUDE_CMD, "--model", config.CLAUDE_MODEL, "-p", prompt, "--output-format", "json"]
    if state.claude_session_id:
        cmd.extend(["--resume", state.claude_session_id])

    try:
        raw = await run_cli(cmd, timeout=timeout)
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"

    try:
        data = json.loads(raw)
        session_id = data.get("session_id") or data.get("sessionId")
        if session_id:
            state.claude_session_id = session_id

        result = data.get("result", "")
        if not result:
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
        return result or raw
    except (json.JSONDecodeError, TypeError):
        return raw
