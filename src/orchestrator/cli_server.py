"""MCP server (Option A) — Cursor delegates to Claude Code and Gemini CLI.

Cursor handles ~95% of tasks directly. This server gives it escape hatches:

Gemini tools (research, strong at exploration, weak at code):
    - research()  → deep domain/technology exploration
    - explain()   → explain unfamiliar code, concepts, or patterns
    - compare()   → research trade-offs between approaches

Claude tools (architecture + implementation, full codebase access):
    - architect()  → design plans, multi-file coordination
    - implement()  → complex implementation Cursor can't handle
    - review()     → thorough code review before PRs
    - debug()      → root cause analysis from errors/stack traces
    - test()       → generate test cases from specs or functions
    - document()   → generate documentation from code

Usage tools:
    - claude_usage() → token counts and costs from Claude Code sessions
    - gemini_usage() → call count and uptime from this server session

Both CLIs run as subprocesses from the project root, so they get native
codebase access without custom filesystem tools.
"""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

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


# Active session IDs — enable --resume across calls within one MCP server
# lifetime. First call creates a session, subsequent calls continue it so
# the model remembers prior context (file reads, decisions).
_claude_session_id: str | None = None
_gemini_session_id: str | None = None


async def _run_gemini(prompt: str) -> str:
    """Run a prompt through Gemini CLI via npx, continuing the session if one exists."""
    global _gemini_session_id

    if not _cli_available(NPX_CMD):
        return f"Error: npx not found at `{NPX_CMD}`. Set NPX_CMD env var or check your nvm install."

    cmd = [NPX_CMD, "-y", GEMINI_PKG, "-m", GEMINI_MODEL, "-p", prompt, "-o", "json"]

    # Continue existing session if we have one
    if _gemini_session_id:
        cmd.extend(["--resume", _gemini_session_id])

    try:
        raw = await _run_cli(cmd)
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"

    # Parse JSON output to extract session ID and response text
    try:
        data = json.loads(raw)
        session_id = data.get("session_id") or data.get("sessionId")
        if session_id:
            _gemini_session_id = session_id

        # Extract text result
        result = data.get("result", "") or data.get("response", "") or data.get("text", "")
        if not result:
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
        return result or raw
    except (json.JSONDecodeError, TypeError):
        return raw


async def _run_claude(prompt: str) -> str:
    """Run a prompt through Claude Code CLI, continuing the session if one exists."""
    global _claude_session_id

    if not _cli_available(CLAUDE_CMD):
        return f"Error: Claude CLI not found at `{CLAUDE_CMD}`. Set CLAUDE_CMD env var or install Claude Code."

    cmd = [CLAUDE_CMD, "--model", CLAUDE_MODEL, "-p", prompt, "--output-format", "json"]

    # Continue existing session if we have one
    if _claude_session_id:
        cmd.extend(["--resume", _claude_session_id])

    try:
        raw = await _run_cli(cmd)
    except (TimeoutError, RuntimeError) as e:
        return f"Error: {e}"

    # Parse JSON output to extract session ID and response text
    try:
        data = json.loads(raw)
        # Capture session ID for future --resume calls
        session_id = data.get("session_id") or data.get("sessionId")
        if session_id:
            _claude_session_id = session_id

        # Extract the text result
        result = data.get("result", "")
        if not result:
            # Fallback: try to get text from content blocks
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
        return result or raw
    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, return raw output
        return raw


def _build_prompt(*parts: str) -> str:
    """Join non-empty prompt parts with double newlines."""
    return "\n\n".join(p for p in parts if p)


# --- Gemini Tools (research, exploration, analysis) ---


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
    prompt = _build_prompt(
        question,
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("gemini", "research")
    return await _run_gemini(prompt)


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
    prompt = _build_prompt(
        "Explain the following clearly and thoroughly. Break down how it works, "
        "why it's done this way, and call out any non-obvious behavior or gotchas.\n",
        code_or_concept,
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("gemini", "explain")
    return await _run_gemini(prompt)


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
    prompt = _build_prompt(
        "Compare the following two options. For each, cover: strengths, weaknesses, "
        "trade-offs, and when to prefer one over the other. End with a recommendation.\n",
        f"## Option A\n\n{option_a}",
        f"## Option B\n\n{option_b}",
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("gemini", "compare")
    return await _run_gemini(prompt)


# --- Claude Tools (architecture, implementation, code quality) ---


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
    prompt = _build_prompt(
        "You are a senior software architect. Design a detailed implementation plan "
        "with specific file paths, function names, and step-by-step changes.\n",
        goal,
        f"## Context\n\n{context}" if context else "",
        f"## Constraints\n\n{constraints}" if constraints else "",
    )
    _track_call("claude", "architect")
    return await _run_claude(prompt)


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
    prompt = _build_prompt(
        spec,
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("claude", "implement")
    return await _run_claude(prompt)


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
    prompt = _build_prompt(
        "You are a senior code reviewer. Review the following thoroughly. "
        "Check for: bugs, security vulnerabilities, performance issues, "
        "readability problems, and missed edge cases. Be specific — reference "
        "line numbers and suggest concrete fixes.\n",
        target,
        f"## Focus area\n\n{focus}" if focus else "",
    )
    _track_call("claude", "review")
    return await _run_claude(prompt)


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
    prompt = _build_prompt(
        "You are debugging an issue. Analyze the error below, read the relevant "
        "source files, trace the execution path, and identify the root cause. "
        "Provide a clear explanation and a concrete fix.\n",
        f"## Error\n\n{error}",
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("claude", "debug")
    return await _run_claude(prompt)


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
    prompt = _build_prompt(
        "Generate comprehensive test cases for the following. Cover: happy paths, "
        "edge cases, error scenarios, and boundary conditions. Use the project's "
        "existing test framework and patterns if any exist.\n",
        target,
        f"## Context\n\n{context}" if context else "",
    )
    _track_call("claude", "test")
    return await _run_claude(prompt)


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
    prompt = _build_prompt(
        "Generate clear, accurate documentation for the following. Read the source "
        "code to understand the actual behavior — don't guess or hallucinate.\n",
        target,
        f"## Style\n\n{style}" if style else "",
    )
    _track_call("claude", "document")
    return await _run_claude(prompt)


@mcp.tool()
async def new_session() -> str:
    """Reset both Claude and Gemini sessions. Next call to either model
    starts a fresh conversation with no prior context.

    Use when switching to a completely different task and you don't want
    prior context bleeding in.
    """
    global _claude_session_id, _gemini_session_id
    cleared = []
    if _claude_session_id:
        cleared.append(f"Claude (`{_claude_session_id}`)")
        _claude_session_id = None
    if _gemini_session_id:
        cleared.append(f"Gemini (`{_gemini_session_id}`)")
        _gemini_session_id = None

    if cleared:
        return f"Sessions cleared: {', '.join(cleared)}. Next calls start fresh."
    return "No active sessions. Next calls start fresh."


# --- Session tracking for usage stats ---

_session_stats = {
    "start_time": time.time(),
    "claude_calls": 0,
    "gemini_calls": 0,
    "claude_tools": {},  # tool_name → count
    "gemini_tools": {},  # tool_name → count
}


def _track_call(model: str, tool_name: str):
    """Track a tool call for usage reporting."""
    key = f"{model}_calls"
    _session_stats[key] = _session_stats.get(key, 0) + 1
    tools_key = f"{model}_tools"
    _session_stats[tools_key][tool_name] = _session_stats[tools_key].get(tool_name, 0) + 1


# --- Usage Tools ---


@mcp.tool()
async def claude_usage() -> str:
    """Show Claude Code usage — token counts and costs from recent sessions.

    Reads Claude Code's session logs to show how many tokens were used
    and approximate costs. Shows both the current MCP server session stats
    and historical Claude Code session data.
    """
    parts: list[str] = []

    # --- MCP server session stats ---
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

    # --- Claude Code session logs ---
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        parts.append("\n## Historical Usage\n\nNo Claude Code session data found.")
        return "\n".join(parts)

    # Find recent session files (last 7 days)
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

    # Approximate costs (Opus pricing: $15/M input, $75/M output, $1.875/M cache read)
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
    uptime_s = time.time() - _session_stats["start_time"]
    uptime_m = int(uptime_s // 60)

    parts = [
        f"## Gemini Usage (this session)\n\n"
        f"**Uptime:** {uptime_m}m\n"
        f"**Model:** {GEMINI_MODEL}\n"
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


def main():
    """Entry point — run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
