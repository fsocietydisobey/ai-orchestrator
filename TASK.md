# Task: Refactor cli_server.py into a Modular Architecture

## Assessment

**Current state:** `src/orchestrator/cli_server.py` is a single ~853-line file that implements the "Option A" MCP server: Cursor delegates to Claude Code and Gemini CLIs via subprocesses. The file mixes:

- **Configuration** — env vars (PROJECT_ROOT, CLI_TIMEOUT, models, CLI paths, ORCHESTRATE_TIMEOUT)
- **Low-level helpers** — `_cli_available`, `_run_cli`, `_run_gemini`, `_run_claude`, `_build_prompt`
- **Session state** — `_claude_session_id`, `_gemini_session_id`, `_session_stats`, `_track_call`
- **Orchestrate internals** — `_OPEN_QUESTION_PATTERNS`, `_has_open_questions`, `_orchestrate_pipeline`
- **13 MCP tools** — research, explain, compare; architect, implement, review, debug, test, document; orchestrate; new_session; claude_usage, gemini_usage

**Why refactor:** Single-file layout makes it harder to test helpers in isolation, add or change tools without scrolling a large file, and reason about dependencies. A modular layout improves maintainability and keeps the same external behavior and entry point.

**Success criteria:**

- Same public behavior: same MCP server name, same tools and signatures, same env-based config.
- Entry point preserved: `python -m orchestrator.cli_server` or existing script still works.
- Clear separation: config, CLI/session helpers, and tool definitions live in dedicated modules.
- No breaking changes for Cursor or existing docs (e.g. docs/CONTEXT.md, docs/mcp-tools-guide.md).

---

## Research

**Relevant patterns:**

- **Config in one place:** Centralize env reads in a small `config` module so CLI paths, timeouts, and model names are easy to override and test.
- **Thin server file:** The FastMCP app and tool registration can live in a single small module that imports and registers tools from submodules; this keeps the “wiring” visible in one place.
- **Helpers by concern:** CLI execution (`_run_cli`, `_cli_available`) and prompt building (`_build_prompt`) are independent of MCP; session runners (`_run_gemini`, `_run_claude`) depend on config and CLI helpers. Splitting into `cli`, `prompts`, and `sessions` (or `runners`) keeps dependencies acyclic.
- **Tools by family:** Group Gemini tools, Claude tools, orchestrate, session/usage tools into separate modules; each module receives the shared `mcp` instance (or a way to register) and the helpers it needs (e.g. `_run_gemini`, `_run_claude`, `_build_prompt`, session state). This mirrors the existing logical grouping in the docstring.

**Risks:**

- **Circular imports:** Avoid by having a single place that creates the FastMCP app and passes it (or a registry) into tool modules that only register handlers. Session state and stats can live in a small `state` or `sessions` module imported by both runners and usage tools.
- **Entry point:** Preserve the current way the server is run (e.g. `uv run python -m orchestrator.cli_server` or similar). Use a thin `cli_server/__main__.py` or keep a one-line `cli_server.py` that delegates to the new package.

---

## Architecture

**Proposed layout:**

```
src/orchestrator/
  cli_server.py              # Thin entry point: from .cli_server_pkg import main; main()
  cli_server_pkg/            # New package (or name it "mcp_cli" if preferred)
    __init__.py              # Exports: mcp, main (and whatever is needed for entry point)
    config.py                # PROJECT_ROOT, CLI_TIMEOUT, CLAUDE_*, GEMINI_*, NPX_CMD, ORCHESTRATE_TIMEOUT
    helpers/
      __init__.py
      cli.py                 # cli_available(), run_cli()
      prompts.py             # build_prompt()
    session/
      __init__.py
      state.py               # Session IDs + _session_stats, track_call()
      runners.py             # run_gemini(), run_claude() — use config + helpers.cli + state
    tools/
      __init__.py             # register_all(mcp) — imports and calls register functions
      gemini.py               # register_gemini_tools(mcp) — research, explain, compare
      claude.py               # register_claude_tools(mcp) — architect, implement, review, debug, test, document
      orchestrate.py          # register_orchestrate(mcp) — orchestrate + pipeline + _has_open_questions
      session_usage.py        # register_session_usage_tools(mcp) — new_session, claude_usage, gemini_usage
    server.py                # FastMCP("ai-orchestrator"), import config, call tools.register_all(mcp)
```

**Data flow:**

- `config` is imported by `helpers.cli`, `session.runners`, and `tools` as needed.
- `session.state` holds session IDs and `_session_stats`; `session.runners` and `tools.session_usage` import state.
- `helpers.cli` and `helpers.prompts` have no dependency on MCP or session state.
- `server.py` creates `mcp`, calls `tools.register_all(mcp)`, and exposes `main()` that runs `mcp.run()`.
- Entry point `cli_server.py` stays at package root for backward compatibility: `from orchestrator.cli_server_pkg import main; main()` (or equivalent so `python -m orchestrator.cli_server` still works).

**Design decisions:**

- **Package name `cli_server_pkg`:** Keeps the top-level name `cli_server` for the entry module; the package holds the implementation. Alternative: rename to `mcp_cli` and have `cli_server.py` import from `mcp_cli` if you want the package name to reflect “MCP via CLI” rather than “server.”
- **Session state in one module:** Avoids spreading globals across files; `runners` and `session_usage` both use the same state module.
- **Orchestrate in one module:** `_orchestrate_pipeline`, `_has_open_questions`, and `_OPEN_QUESTION_PATTERNS` stay together in `tools/orchestrate.py` since they are only used by the orchestrate tool.
- **Registration pattern:** Each tool module exposes `register_*(mcp)` so `server.py` can wire everything in one place without importing individual tool functions.

---

## Open Questions (Resolved)

- **Where does FastMCP live?** In `server.py`; `tools` receive `mcp` and register handlers. No circular import because tools do not import `server`.
- **How does entry point work?** Keep `src/orchestrator/cli_server.py` as a one-liner that imports and calls `main()` from the new package, so existing `python -m orchestrator.cli_server` and any scripts/docs referring to it remain valid.

---

## Implementation Plan

1. **Create package and config**
   - Add `src/orchestrator/cli_server_pkg/` with `__init__.py`.
   - Add `config.py`: move all env-based constants from `cli_server.py` (PROJECT_ROOT, CLI_TIMEOUT, CLAUDE_MODEL, GEMINI_MODEL, CLAUDE_CMD, NPX_CMD, GEMINI_PKG, ORCHESTRATE_TIMEOUT). Export them for use by helpers and tools.

2. **Extract helpers**
   - Add `helpers/__init__.py`, `helpers/cli.py`, `helpers/prompts.py`.
   - Move `_cli_available` → `cli.cli_available`, `_run_cli` → `cli.run_cli` (use `config.PROJECT_ROOT`, `config.CLI_TIMEOUT`).
   - Move `_build_prompt` → `prompts.build_prompt`.
   - Keep function behavior identical (same signatures and return values).

3. **Extract session state and runners**
   - Add `session/__init__.py`, `session/state.py`, `session/runners.py`.
   - In `state.py`: move `_claude_session_id`, `_gemini_session_id`, `_session_stats`, `_track_call`; expose getters/setters or module-level variables as needed by runners and usage tools.
   - In `runners.py`: implement `run_gemini(prompt)` and `run_claude(prompt)` using config, `helpers.cli`, and `session.state` (session IDs, track_call). Move JSON parsing and session-id extraction from current `_run_gemini`/`_run_claude` into these functions.

4. **Extract orchestrate helpers**
   - Add `tools/orchestrate.py`. Move `_OPEN_QUESTION_PATTERNS`, `_has_open_questions`, and the full `_orchestrate_pipeline` and `orchestrate` tool implementation. Use `session.runners`, `session.state`, `helpers.prompts`, and `config` (ORCHESTRATE_TIMEOUT, PROJECT_ROOT). Register the orchestrate tool with `mcp` via a `register_orchestrate(mcp)` function.

5. **Extract Gemini and Claude tools**
   - Add `tools/gemini.py`: move research, explain, compare; call `runners.run_gemini` and `state.track_call`; expose `register_gemini_tools(mcp)`.
   - Add `tools/claude.py`: move architect, implement, review, debug, test, document; call `runners.run_claude` and `state.track_call`; expose `register_claude_tools(mcp)`.

6. **Extract session and usage tools**
   - Add `tools/session_usage.py`: move new_session, claude_usage, gemini_usage; use `session.state` and config (e.g. GEMINI_MODEL for gemini_usage); expose `register_session_usage_tools(mcp)`.

7. **Wire server and entry point**
   - Add `server.py`: create `mcp = FastMCP("ai-orchestrator")`, then call `register_gemini_tools(mcp)`, `register_claude_tools(mcp)`, `register_orchestrate(mcp)`, `register_session_usage_tools(mcp)`. Define `main()` that runs `mcp.run()`.
   - In `cli_server_pkg/__init__.py`, export `mcp` and `main` from `server`.
   - Replace the body of `src/orchestrator/cli_server.py` with a thin wrapper that imports `main` from the new package and runs it (e.g. `from orchestrator.cli_server_pkg import main; main()`), preserving the module docstring at the top of `cli_server.py` so docs and IDE still see the high-level description.

8. **Verify**
   - Run the MCP server (e.g. `uv run python -m orchestrator.cli_server`) and confirm it starts without import errors.
   - If the project has tests that invoke the server or tools, run them and fix any import or registration issues.
   - Optionally add a minimal test that imports `orchestrator.cli_server_pkg` and calls `main` or checks that `mcp` has the expected tools.

---

## Summary

Refactoring splits the monolith into: **config**, **helpers** (cli, prompts), **session** (state + runners), and **tools** (gemini, claude, orchestrate, session_usage), with a **server** module that wires FastMCP and a **thin cli_server.py** entry point. Behavior and entry point stay the same; only the internal layout and separation of concerns change.
