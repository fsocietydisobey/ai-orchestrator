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

Orchestration:
    - orchestrate() → multi-model deliberation: Gemini researches + Claude architects
                       in a back-and-forth loop, producing TASK.md

Usage tools:
    - claude_usage() → token counts and costs from Claude Code sessions
    - gemini_usage() → call count and uptime from this server session

Both CLIs run as subprocesses from the project root, so they get native
codebase access without custom filesystem tools.

Implementation lives in the cli_server_pkg package (modular layout).
"""

from orchestrator.cli_server_pkg import main

if __name__ == "__main__":
    main()
