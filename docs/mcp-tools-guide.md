# MCP Tools Guide

How to use the ai-orchestrator MCP tools from Cursor (or Claude Code).

## Overview

The MCP server gives your IDE access to 12 tools backed by two AI models:

- **Gemini** (gemini-3.1-pro-preview) — research and analysis. Strong at exploring topics, explaining concepts, and comparing options. Weak at writing code.
- **Claude Code** (claude-opus-4-6) — architecture, implementation, and code quality. Runs with full codebase access via `claude -p`.

Cursor handles most tasks directly. These tools are for when you need a different model's strengths.

## Quick start

### Keyword shortcuts

Prefix your message with a model name to force routing:

```
gemini, how do WebSocket reconnection strategies work?
claude, review the auth middleware for security issues
```

Or use action keywords:

```
research how LangGraph checkpointers work
explain what create_react_agent does under the hood
compare Redis vs Memcached for session storage
plan a caching layer for our API
debug this TypeError: Cannot read property 'id' of undefined
review src/orchestrator/cli_server.py
write tests for the router module
document the MCP server tools
new session — switching to a different task
show claude usage
show gemini usage
```

## Tools reference

### Gemini tools

These route to Gemini. Use for thinking and understanding, never for writing code.

---

#### `research(question, context?)`

Deep exploration of a topic, technology, or domain.

**When to use**: Before building something you don't fully understand. When you need to explore options, understand trade-offs, or investigate how something works.

**Examples**:

```
research how LangGraph checkpointers work and when to use MemorySaver vs PostgresSaver

research what are the best practices for MCP server error handling

gemini, investigate how confidence-gated human-in-the-loop pipelines work
```

With context:

```
research how to add rate limiting to a FastAPI app
  context: we're using Redis already for caching, want to reuse the connection
```

---

#### `explain(code_or_concept, context?)`

Explain unfamiliar code, patterns, or concepts.

**When to use**: When you encounter something you don't understand — a complex function, library pattern, design pattern, or concept.

**Examples**:

```
explain what this decorator pattern does: @tool decorator from langchain_core

explain the add_messages reducer in LangGraph

gemini, explain how Python's asyncio.create_subprocess_exec works
```

With context:

```
explain the route_after_classify function
  context: I'm trying to understand how the LangGraph pipeline decides which node to visit next
```

---

#### `compare(option_a, option_b, context?)`

Compare two approaches, technologies, or options with trade-offs.

**When to use**: When deciding between alternatives. Libraries, patterns, architectures, strategies.

**Examples**:

```
compare FastAPI vs Flask for building MCP servers

compare subprocess.run vs asyncio.create_subprocess_exec for CLI wrappers

gemini, compare LangGraph vs CrewAI for multi-agent orchestration
```

With context:

```
compare SQLAlchemy vs Prisma
  context: Python backend, PostgreSQL, team is familiar with Django ORM
```

---

### Claude tools

These route to Claude Code (Opus 4.6). Claude runs with full codebase access — it can read files, search code, and understand your project natively.

---

#### `architect(goal, context?, constraints?)`

Design an implementation plan with specific file paths and step-by-step changes.

**When to use**: When the problem is understood but the solution needs design. Multi-file changes, API design, system architecture.

**Examples**:

```
plan a retry mechanism for failed API calls with exponential backoff

architect a caching layer for the research tool results

claude, design a plugin system for adding new MCP tools dynamically
```

With constraints:

```
architect a WebSocket notification system
  context: users need real-time updates when background tasks complete
  constraints: must work with existing FastAPI backend, no new dependencies
```

---

#### `implement(spec, context?)`

Implement complex tasks that Cursor can't handle well on its own.

**When to use**: Intricate refactors, multi-file changes with dependencies, tasks that need deep codebase analysis before writing code. For simple tasks, let Cursor handle it directly.

**Examples**:

```
claude, implement the retry mechanism from the architecture plan above

implement a CLI subprocess provider that wraps both claude and gemini with timeout and error handling

claude, refactor the provider classes to support streaming responses
```

---

#### `review(target, focus?)`

Thorough code review covering bugs, security, performance, and readability.

**When to use**: Before PRs. When you want a second opinion. When reviewing unfamiliar code.

**Examples**:

```
review src/orchestrator/cli_server.py

review the recent changes to the router module, focus on security

claude, review the filesystem tools for path traversal vulnerabilities
```

With focus:

```
review src/orchestrator/providers/google_provider.py
  focus: error handling and edge cases
```

---

#### `debug(error, context?)`

Root cause analysis from error messages, stack traces, or unexpected behavior.

**When to use**: When you have an error you can't figure out. When you need to trace execution across multiple files.

**Examples**:

```
debug TypeError: 'NoneType' object is not subscriptable at router.py:78

debug the MCP server crashes on startup with "No such file or directory"

claude, debug why the classify node returns invalid JSON intermittently
```

With context:

```
debug ConnectionRefusedError when calling the research tool
  context: just upgraded the google-genai package, was working before
```

---

#### `test(target, context?)`

Generate comprehensive test cases covering happy paths, edge cases, and error scenarios.

**When to use**: When you need tests for existing code. When you want thorough coverage of a function or module.

**Examples**:

```
write tests for the _run_cli helper function

test the classify node with various task descriptions

claude, generate tests for the Router class
```

With context:

```
test src/orchestrator/tools/filesystem.py
  context: use pytest, mock the filesystem calls
```

---

#### `document(target, style?)`

Generate documentation from source code.

**When to use**: When you need docs for a module, API, or feature. When you want docstrings, a readme section, or an API reference.

**Examples**:

```
document the MCP server tools

document src/orchestrator/config.py

claude, write docs for the provider interface
```

With style:

```
document the cli_server module
  style: API reference with usage examples for each tool
```

---

### Session management

Claude Code calls are session-persistent — the first Claude call creates a session, and every subsequent call continues it using `--resume`. This means Claude remembers what files it read, what plans it made, and what context it has across multiple tool calls.

---

#### `new_session()`

Reset the Claude session. Next Claude call starts a fresh conversation with no prior context.

**When to use**: When switching to a completely different task and you don't want prior context bleeding in.

**Examples**:

```
new session — I'm done with the retry work, starting on auth now

clear the claude session

start fresh
```

**How sessions work**:
- First Claude tool call (architect, implement, review, etc.) creates a new session
- All subsequent Claude calls continue that same session — Claude remembers everything
- `new_session` clears it so the next call starts fresh
- If the MCP server restarts (e.g. Cursor reload), the session resets automatically
- All Cursor chats share one Claude session — use `new_session` when switching tasks

---

### Usage tools

---

#### `claude_usage()`

Show Claude Code usage — MCP session call counts plus historical token usage from the last 7 days of Claude Code sessions.

**Examples**:

```
show claude usage

how much claude have I used this week
```

---

#### `gemini_usage()`

Show Gemini usage for the current MCP server session — call counts and breakdown by tool.

**Examples**:

```
show gemini usage

how many gemini calls have I made
```

*Gemini CLI doesn't store local usage logs. Check Google AI Studio for detailed billing.*

---

## Chaining tools

The real power is using tools together in a single Cursor conversation. Claude session persistence means each step builds on the last — Claude remembers what it read and decided:

```
1. "research how retry mechanisms work with exponential backoff and jitter"
2. "architect a retry system for our API calls based on those findings"
3. "implement that plan"          ← Claude already knows the plan from step 2
4. "write tests for the retry logic"  ← Claude already knows the code from step 3
5. "review the implementation for edge cases"
```

Or mix models:

```
1. "gemini, research the trade-offs of event sourcing vs CRUD for our use case"
2. "claude, architect a solution based on that research"
3. "implement it"                 ← Claude continues from step 2
```

When switching to a different task:

```
"new session — now let's work on the auth system"
```

## Configuration

### Basic setup (recommended)

No env vars needed — defaults are baked into `cli_server.py`:

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/_3ntropy/dev/ai-orchestrator",
        "run",
        "ai-orchestrator"
      ]
    }
  }
}
```

Defaults: Claude Opus 4.6, Gemini 3.1 Pro Preview, 5 min timeout, nvm v24.12.0.

### Optional overrides

Add `env` only if you need to override defaults on a specific machine (different nvm version, different model, longer timeout, etc.):

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/_3ntropy/dev/ai-orchestrator",
        "run",
        "ai-orchestrator"
      ],
      "env": {
        "CLAUDE_MODEL": "claude-sonnet-4-6",
        "CLI_TIMEOUT": "600"
      }
    }
  }
}
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model for architect/implement/review/debug/test/document |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | Gemini model for research/explain/compare |
| `CLAUDE_CMD` | `claude` | Path to Claude Code CLI |
| `NPX_CMD` | `~/.nvm/versions/node/v24.12.0/bin/npx` | Path to npx (for Gemini CLI) |
| `GEMINI_PKG` | `@google/gemini-cli@latest` | npm package for Gemini CLI |
| `NVM_NODE_VERSION` | `24.12.0` | nvm Node version (used to resolve npx path) |
| `CLI_TIMEOUT` | `300` | Max seconds per CLI call before timeout |
| `PROJECT_ROOT` | current working directory | Root directory CLIs run from |

## Troubleshooting

**Tools don't show up in Cursor**: Restart the MCP server in Cursor Settings > MCP. Check for a green status indicator.

**Cursor uses web search instead of `research`**: Be explicit — say "use the research MCP tool" or prefix with "gemini,". The `.cursor/rules` file helps with automatic routing.

**Gemini errors with "npx not found"**: The server needs the absolute path to npx from your nvm install. Check that `NVM_NODE_VERSION` matches your installed version.

**Claude errors with "CLI not found"**: Make sure Claude Code is installed and `claude` is on your PATH.

**Timeout errors**: Increase `CLI_TIMEOUT` in env vars. Default is 5 minutes (300s). Complex tasks may need longer.

**First Gemini call is slow**: `npx` downloads the package on first run. Subsequent calls use the cached version.
