# Issues & Improvements

Identified during codebase review (2026-03-14). Organized by priority.

## High Priority

### No tests
There are no test files despite a `testing-guide.md`. This is the biggest gap for a system orchestrating multiple AI models with complex state flows.
- [ ] Unit tests for state reducers and graph routing logic (`select_next_node`, `_research_exit`)
- [ ] Unit tests for `Router.classify()` JSON parsing and fallback behavior
- [ ] Integration tests for CLI subprocess runners (`run_claude`, `run_gemini`)
- [ ] End-to-end tests for the LangGraph pipeline (mock CLI outputs, verify state transitions)
- [ ] Tests for MCP tool handlers (both Option A and Option B)

### No error recovery in graph
If a CLI subprocess fails mid-graph, there's no retry or fallback. The supervisor doesn't handle node failures.
- [ ] Add try/except in domain nodes (research, architect, implement) that write failure info to state
- [ ] Teach supervisor to detect node failures and decide whether to retry, skip, or abort
- [ ] Add max retry limits per node to prevent infinite loops

### Subprocess input sanitization
`cli_server_pkg/session/runners.py` passes user input into CLI commands via subprocess.
- [ ] Audit all subprocess calls for injection risks
- [ ] Ensure arguments are passed as list elements (not shell-interpolated strings)
- [ ] Add input validation at MCP tool boundaries

## Medium Priority

### No logging framework
Debugging multi-model orchestration across subprocesses is difficult without structured logging.
- [ ] Add `logging` or `structlog` with per-node context (node name, thread_id, attempt number)
- [ ] Log supervisor decisions, node inputs/outputs, and timing
- [ ] Add log levels so production can filter noise

### No rate limiting
Concurrent tool calls (especially fan-out research) could hit API rate limits.
- [ ] Add rate limiting or semaphore for concurrent CLI subprocess calls
- [ ] Handle rate-limit errors from API providers gracefully (backoff + retry)

### Persistent checkpointing
`InMemorySaver` loses all state on restart.
- [ ] Switch to `SqliteSaver` or `AsyncSqliteSaver` for development
- [ ] Consider Redis or Postgres saver for production deployments

### No type checking setup
Type hints are used throughout but there's no mypy/pyright configuration.
- [ ] Add `mypy` or `pyright` config to `pyproject.toml`
- [ ] Add type checking to CI/pre-commit

## Low Priority

### Heavy dependency tree
Option A barely uses LangGraph/LangChain but pulls them in via shared package.
- [ ] Consider making Option B dependencies optional extras in `pyproject.toml`
- [ ] e.g., `pip install ai-orchestrator[graph]` for LangGraph support

### TASK.md committed to repo
`TASK.md` is generated output from the `orchestrate` tool — ephemeral by nature.
- [ ] Add `TASK.md` to `.gitignore`

### API keys in MCP config
Option B requires API keys in `~/.cursor/mcp.json`, which users might accidentally commit.
- [ ] Document the `.env` approach as the recommended method
- [ ] Add a warning in the MCP tool if keys are detected in config rather than env
