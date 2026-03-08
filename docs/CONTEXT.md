# AI Orchestrator — Context for AI Models

This document explains the ai-orchestrator project so any AI model (Claude, Gemini, Cursor, etc.) can understand what it is, how it works, and how to use it. Include this file or its contents when working on this project in a new environment.

## What this is

An MCP (Model Context Protocol) server that gives Cursor IDE access to Claude Code and Gemini CLI as tools. Instead of being limited to Cursor's built-in AI model, you can delegate tasks to the right model based on what it's best at.

**The core idea**: each AI model has different strengths. Rather than using one model for everything, route tasks to the model that's best at that type of work.

## Why it exists

Cursor is great at implementing ~95% of coding tasks. But it falls short on:

1. **Deep research** — exploring unfamiliar domains, technologies, or trade-offs before building
2. **Complex architecture** — multi-file coordination, intricate refactors, tasks needing deep codebase analysis
3. **Complex implementation** — tasks where Cursor's built-in models aren't strong enough

This MCP server solves that by giving Cursor two escape hatches via CLI subprocesses:

- **Gemini CLI** (`gemini -p "..."`) — strong researcher, weak coder. Used for exploration and analysis only.
- **Claude Code CLI** (`claude -p "..."`) — strong architect and implementer. Runs with full codebase access (file reading, grep, glob, editing) natively.

Both CLIs run from the project root directory, so they get native codebase context without needing custom filesystem tools.

## Architecture

```
You (working in Cursor IDE)
  │
  ├── Simple tasks → Cursor handles directly (95% of work)
  │
  └── Complex tasks → MCP tool call → ai-orchestrator server
                                        │
                                        ├── research / explain / compare
                                        │     → Gemini CLI subprocess
                                        │     → gemini-3.1-pro-preview
                                        │
                                        └── architect / implement / review / debug / test / document
                                              → Claude Code CLI subprocess
                                              → claude-opus-4-6
```

The server is a single Python file (`src/orchestrator/cli_server.py`) using FastMCP. It spawns CLI subprocesses for each tool call, captures the output, and returns it to Cursor.

## Session persistence

Both Claude and Gemini calls are session-persistent within one MCP server lifetime:

- First tool call creates a new CLI session
- Subsequent calls use `--resume <session_id>` to continue the same conversation
- The model remembers what files it read, what plans it made, and what context it has
- `new_session` tool resets both sessions for a fresh start

**Limitation**: all Cursor chats share one Claude session and one Gemini session (MCP doesn't pass conversation IDs from the client). Use `new_session` when switching between unrelated tasks.

## The 12 tools

### Gemini tools (research and analysis — never writes code)

| Tool | Purpose | Trigger keywords |
|------|---------|-----------------|
| `research(question, context?)` | Deep domain/technology exploration | "research", "deep dive", "investigate" |
| `explain(code_or_concept, context?)` | Explain unfamiliar code or concepts | "explain", "what is", "how does ... work" |
| `compare(option_a, option_b, context?)` | Trade-off analysis between options | "compare", "... vs ...", "trade-offs" |

### Claude tools (architecture, implementation, code quality — full codebase access)

| Tool | Purpose | Trigger keywords |
|------|---------|-----------------|
| `architect(goal, context?, constraints?)` | Design implementation plans | "plan", "design", "architect" |
| `implement(spec, context?)` | Complex implementation Cursor can't handle | "claude, implement..." |
| `review(target, focus?)` | Thorough code review | "review", "code review" |
| `debug(error, context?)` | Root cause analysis | "debug", "why is ... failing" |
| `test(target, context?)` | Generate test cases | "test", "write tests for" |
| `document(target, style?)` | Generate documentation | "document", "write docs for" |

### Session and usage tools

| Tool | Purpose | Trigger keywords |
|------|---------|-----------------|
| `new_session()` | Reset Claude + Gemini sessions | "new session", "start fresh" |
| `claude_usage()` | Token counts + costs (7-day history) | "show claude usage" |
| `gemini_usage()` | Call counts for current session | "show gemini usage" |

### Keyword shortcuts

Prefix any message with the model name to force routing:

- `"gemini, ..."` → routes to the appropriate Gemini tool
- `"claude, ..."` → routes to the appropriate Claude tool

## Typical workflow

```
1. "research how retry mechanisms work with exponential backoff"
     → Gemini researches, returns findings

2. "architect a retry system for our API calls based on those findings"
     → Claude reads the codebase, designs a plan (continues session from step 1)

3. "implement that plan"
     → Claude implements it (continues session, already knows the plan + codebase)

4. "write tests for the retry logic"
     → Claude generates tests (continues session, knows the implementation)

5. "review the implementation for edge cases"
     → Claude reviews (continues session, has full context)

6. "new session — switching to auth work now"
     → Clears both sessions for a clean slate
```

## Project structure

```
ai-orchestrator/
├── src/orchestrator/
│   ├── cli_server.py          ← THE MAIN FILE — MCP server with all 12 tools
│   ├── server.py              ← LangGraph server (experimental, Option B)
│   ├── config.py              ← Loads config.yaml (used by LangGraph server)
│   ├── models.py              ← LangChain model factories (used by LangGraph server)
│   ├── state.py               ← OrchestratorState TypedDict (used by LangGraph server)
│   ├── graph.py               ← StateGraph construction (used by LangGraph server)
│   ├── router.py              ← Role → provider routing (used by LangGraph server)
│   ├── nodes/                 ← LangGraph node factories (experimental)
│   ├── tools/                 ← Custom filesystem tools (used by LangGraph server)
│   ├── providers/             ← Raw API providers (used by LangGraph server)
│   └── prompts/               ← System prompts for each role
├── .cursor/rules              ← Cursor routing rules (keyword → MCP tool)
├── docs/
│   ├── mcp-tools-guide.md     ← Detailed usage guide for all tools
│   └── CONTEXT.md             ← This file
├── config.yaml                ← Model/role config (used by LangGraph server)
├── pyproject.toml             ← Entry points: ai-orchestrator, ai-orchestrator-graph
└── .env.example               ← API key template (only needed for LangGraph server)
```

**Two entry points** in `pyproject.toml`:
- `ai-orchestrator` → `cli_server.py` (Option A — daily driver, CLI subprocesses)
- `ai-orchestrator-graph` → `server.py` (Option B — experimental LangGraph pipeline)

## Setup on a new machine

### Prerequisites

- Python 3.12+ with `uv`
- Claude Code CLI (`claude` on PATH)
- Node.js via nvm (v24.12.0) for Gemini CLI (`npx @google/gemini-cli@latest`)

### Install

```bash
git clone <repo-url>
cd ai-orchestrator
uv sync
```

### Connect to Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/ai-orchestrator",
        "run",
        "ai-orchestrator"
      ]
    }
  }
}
```

No API keys needed — each CLI handles its own auth.

### Connect to Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/ai-orchestrator",
        "run",
        "ai-orchestrator"
      ]
    }
  }
}
```

## Configuration

Defaults are baked into `cli_server.py`. Override via env vars in MCP config only if needed:

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `CLAUDE_MODEL` | `claude-opus-4-6` | Model for all Claude tools |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | Model for all Gemini tools |
| `CLAUDE_CMD` | `claude` | Path to Claude Code CLI |
| `NPX_CMD` | `~/.nvm/versions/node/v24.12.0/bin/npx` | Path to npx (for Gemini CLI) |
| `GEMINI_PKG` | `@google/gemini-cli@latest` | npm package for Gemini CLI |
| `NVM_NODE_VERSION` | `24.12.0` | nvm Node version (resolves npx path) |
| `CLI_TIMEOUT` | `300` | Seconds before subprocess timeout |
| `PROJECT_ROOT` | cwd | Root directory CLIs run from |

## Known limitations

- **Single session per model**: All Cursor chats share one Claude session and one Gemini session. MCP doesn't pass conversation IDs from the client. Use `new_session` when switching tasks.
- **MCP server restart clears sessions**: If Cursor restarts the MCP server (e.g. on reload), sessions reset.
- **First Gemini call is slow**: `npx` downloads the package on first run. Cached after that.
- **nvm dependency**: Gemini CLI requires npx from a specific nvm Node version. The server resolves the absolute path to npx since nvm isn't loaded in subprocess shells.

## Future work (Option B — LangGraph)

The `server.py` + `graph.py` + `nodes/` code is an experimental LangGraph pipeline that routes tasks through classify → research → architect → implement. It uses the same models but through API calls with custom filesystem tools instead of CLI subprocesses. It's a testbed for advanced agent patterns (self-reflection, checkpoints, time-travel). See the README for the full roadmap.
