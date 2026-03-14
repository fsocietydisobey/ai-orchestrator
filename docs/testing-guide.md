# Testing Guide

How to test the ai-orchestrator MCP server from the terminal without connecting to Cursor.

## Prerequisites

```bash
cd /home/_3ntropy/dev/ai-orchestrator
uv sync
```

Make sure `.env` has your API keys (needed for Option B's classify node):

```bash
cat .env
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_AI_API_KEY=AIza...
```

---

## Option A — CLI Server (daily driver)

Option A doesn't need API keys — it shells out to `claude` and `gemini` CLIs which handle their own auth.

### Test individual tools

```bash
# Research (Gemini CLI)
uv run python -c "
import asyncio
from orchestrator.cli_server_pkg.session.runners import run_gemini
print(asyncio.run(run_gemini('What is the MCP protocol?')))
"

# Architect (Claude Code CLI)
uv run python -c "
import asyncio
from orchestrator.cli_server_pkg.session.runners import run_claude
print(asyncio.run(run_claude('Design a health check endpoint for this project')))
"
```

### Test MCP tools directly

```bash
# Research tool
uv run python -c "
import asyncio
from orchestrator.cli_server_pkg.tools.gemini import register_gemini
from orchestrator.cli_server_pkg.server import mcp
# Tools are already registered, call via the module
from orchestrator.cli_server_pkg.session.runners import run_gemini
from orchestrator.cli_server_pkg.helpers.prompts import build_prompt
prompt = build_prompt('What are the best patterns for MCP server design?')
print(asyncio.run(run_gemini(prompt)))
"

# Verify all 13 tools are registered
uv run python -c "
from orchestrator.cli_server_pkg import mcp
print('Tools:', list(mcp._tool_manager._tools.keys()))
"
```

### Test orchestrate tool

```bash
# This takes 5-10 minutes (multiple CLI calls)
# Monitor progress: cat .orchestrate-status
uv run python -c "
import asyncio
from orchestrator.cli_server_pkg.tools.orchestrate import _orchestrate_pipeline
from orchestrator.cli_server_pkg.session.state import reset_sessions
from orchestrator.cli_server_pkg import config
import time

reset_sessions()
start = time.time()
log = []

def check():
    if time.time() - start > config.ORCHESTRATE_TIMEOUT:
        raise TimeoutError('Timed out')

result = asyncio.run(_orchestrate_pipeline(
    'Add a health check endpoint to the MCP server',
    '',
    start,
    3,
    log,
    check,
))
print(result)
"
```

---

## Option B — LangGraph Pipeline

Option B needs `ANTHROPIC_API_KEY` for the classify node (Haiku). Research/architect/implement use CLI subprocesses.

### Test classify (fast, cheap API call)

```bash
uv run python -c "
import asyncio
from orchestrator.server import classify

tasks = [
    'How do LangGraph checkpointers work?',
    'Add retry logic to all API calls',
    'Fix the typo in the header',
]
for t in tasks:
    print(f'\n> {t}')
    print(asyncio.run(classify(t)))
"
```

### Test full chain (classify → research → architect → implement)

```bash
# Research-tier task (full pipeline: research → architect → implement)
uv run python -c "
import asyncio
from orchestrator.server import chain
result = asyncio.run(chain('How do LangGraph checkpointers work and should we use one?'))
print(result)
"

# Architect-tier task (skips research: architect → implement)
uv run python -c "
import asyncio
from orchestrator.server import chain
result = asyncio.run(chain('Add a /health endpoint that returns server uptime and tool count'))
print(result)
"

# Implement-tier task (skips research + architect: implement only)
uv run python -c "
import asyncio
from orchestrator.server import chain
result = asyncio.run(chain('Rename the variable CLI_TIMEOUT to DEFAULT_TIMEOUT in config.py'))
print(result)
"
```

### Test checkpoints and history

```bash
uv run python -c "
import asyncio
from orchestrator.server import chain, history

async def test():
    # Run a chain
    result = await chain('Add a health check endpoint')
    print(result[:500])
    print('---')

    # Extract thread_id
    thread_id = result.split('\`')[1]

    # Show checkpoint history
    h = await history(thread_id)
    print(h)

asyncio.run(test())
"
```

### Test time-travel (rewind)

```bash
uv run python -c "
import asyncio
from orchestrator.server import chain, history, rewind

async def test():
    # Run initial chain
    result = await chain('Add WebSocket support for real-time notifications')
    thread_id = result.split('\`')[1]
    print(f'Thread: {thread_id}')
    print(result[:300])
    print('...\n---')

    # Get history
    h = await history(thread_id)
    print(h)

    # Find the checkpoint after classify (before research)
    # Look for the one where next=research or next=architect
    # Then rewind with different constraints
    print('\n--- Rewinding... ---')

    # Get checkpoint IDs from history
    import re
    ids = re.findall(r'\`([a-f0-9-]+)\.\.\.\`', h)
    if len(ids) >= 2:
        # Rewind to after classify, re-run with different task
        target = ids[-2]  # Second from bottom = after classify
        r = await rewind(
            thread_id,
            target,
            new_task='Add WebSocket support — but use SSE instead, simpler for our use case'
        )
        print(r[:500])

asyncio.run(test())
"
```

### Test self-reflection (critique loop)

The critique loop runs automatically inside `chain()`. To see it in action, check the quality scores in the output:

```bash
uv run python -c "
import asyncio
from orchestrator.server import chain

async def test():
    # Use a vague task — more likely to trigger low scores and retries
    result = await chain('Make the code better')
    print(result)

asyncio.run(test())
"
```

Look for the **Quality Scores** section in the output:
- Score >= 0.7: passed, no retry
- Score < 0.7: retried with critique feedback (up to 2 attempts)

---

## Connecting to Cursor (when ready)

### Option A (recommended)

Add to `~/.cursor/mcp.json`:

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

### Option B (LangGraph)

```json
{
  "mcpServers": {
    "ai-orchestrator-graph": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/_3ntropy/dev/ai-orchestrator",
        "run",
        "ai-orchestrator-graph"
      ]
    }
  }
}
```

No `env` block needed — `.env` is loaded automatically.

### Both at once

You can run both servers simultaneously with different names:

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": ["--directory", "/home/_3ntropy/dev/ai-orchestrator", "run", "ai-orchestrator"]
    },
    "ai-orchestrator-graph": {
      "command": "uv",
      "args": ["--directory", "/home/_3ntropy/dev/ai-orchestrator", "run", "ai-orchestrator-graph"]
    }
  }
}
```

Option A tools: research, explain, compare, architect, implement, review, debug, test, document, orchestrate, new_session, claude_usage, gemini_usage

Option B tools: research, architect, classify, chain, history, rewind
