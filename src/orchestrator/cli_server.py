"""MCP server (Option A) — Cursor delegates to Claude Code and Gemini CLI.

Cursor handles ~95% of tasks directly. This server gives it two escape hatches:
    - research() → Gemini CLI for deep exploration (strong researcher, weak coder)
    - architect() → Claude Code for complex design (full codebase context)
    - implement() → Claude Code for complex implementation Cursor can't handle

Both CLIs run as subprocesses from the project root, so they get native
codebase access without custom filesystem tools.
"""

import asyncio
import os
import shutil

from mcp.server.fastmcp import FastMCP

# --- Configuration ---

# Project root defaults to cwd; override with PROJECT_ROOT env var
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())

# Subprocess timeout in seconds (5 minutes default)
CLI_TIMEOUT = int(os.environ.get("CLI_TIMEOUT", "300"))

# Models — override via env vars
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

# CLI paths — override via env vars if needed
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Gemini runs via npx under nvm — we need the absolute path to npx
# since nvm isn't loaded in subprocess shells
_NVM_NODE_VERSION = os.environ.get("NVM_NODE_VERSION", "24.12.0")
_NVM_BIN = os.path.expanduser(f"~/.nvm/versions/node/v{_NVM_NODE_VERSION}/bin")
NPX_CMD = os.environ.get("NPX_CMD", os.path.join(_NVM_BIN, "npx"))
GEMINI_PKG = os.environ.get("GEMINI_PKG", "@google/gemini-cli@latest")

mcp = FastMCP("ai-orchestrator")


# --- Helpers ---


def _cli_available(cmd: str) -> bool:
    """Check if a CLI tool exists (absolute path or on PATH)."""
    if os.path.isabs(cmd):
        return os.path.isfile(cmd) and os.access(cmd, os.X_OK)
    return shutil.which(cmd) is not None


async def _run_cli(cmd: list[str], timeout: int = CLI_TIMEOUT) -> str:
    """Run a CLI command as a subprocess and return stdout.

    Args:
        cmd: Command and arguments to run.
        timeout: Max seconds to wait before killing the process.

    Returns:
        The process stdout as a string.

    Raises:
        TimeoutError: If the process exceeds the timeout.
        RuntimeError: If the process exits with a non-zero code.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(
            f"CLI command timed out after {timeout}s: {' '.join(cmd[:2])}..."
        )

    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(
            f"CLI exited with code {proc.returncode}: {err or '(no stderr)'}"
        )

    return stdout.decode()


# --- MCP Tools ---


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
    if not _cli_available(NPX_CMD):
        return f"Error: npx not found at `{NPX_CMD}`. Set NPX_CMD env var or check your nvm install."

    prompt = question
    if context:
        prompt = f"{question}\n\n## Context\n\n{context}"

    try:
        result = await _run_cli(
            [NPX_CMD, "-y", GEMINI_PKG, "-m", GEMINI_MODEL, "-p", prompt]
        )
        return result
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"


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
    if not _cli_available(CLAUDE_CMD):
        return f"Error: Claude CLI not found at `{CLAUDE_CMD}`. Set CLAUDE_CMD env var or install Claude Code."

    parts = [
        "You are a senior software architect. Design a detailed implementation plan.\n",
        goal,
    ]
    if context:
        parts.append(f"\n## Context\n\n{context}")
    if constraints:
        parts.append(f"\n## Constraints\n\n{constraints}")
    prompt = "\n".join(parts)

    try:
        result = await _run_cli(
            [CLAUDE_CMD, "--model", CLAUDE_MODEL, "-p", prompt, "--output-format", "text"]
        )
        return result
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"


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
    if not _cli_available(CLAUDE_CMD):
        return f"Error: Claude CLI not found at `{CLAUDE_CMD}`. Set CLAUDE_CMD env var or install Claude Code."

    parts = [spec]
    if context:
        parts.append(f"\n## Context\n\n{context}")
    prompt = "\n".join(parts)

    try:
        result = await _run_cli(
            [CLAUDE_CMD, "--model", CLAUDE_MODEL, "-p", prompt, "--output-format", "text"]
        )
        return result
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"


def main():
    """Entry point — run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
