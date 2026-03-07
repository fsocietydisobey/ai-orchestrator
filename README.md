# AI Orchestrator

LangGraph-powered agent that routes tasks to the right AI model, with full filesystem access for codebase-aware reasoning.

| Role | Model | When to use |
|---|---|---|
| **Research** | Gemini Pro | Domain exploration, technology investigation, understanding unknowns |
| **Architect** | Claude Sonnet/Opus | Design decisions, implementation plans, multi-file coordination |
| **Classify** | Claude Haiku | Fast routing — determines which pipeline a task needs |
| **Implement** | Codex / IDE model | Clear spec exists, ready to write code |

## Architecture

The orchestrator is a **LangGraph StateGraph** exposed via an **MCP server**. Each node in the graph is a different AI model with filesystem tools — so every model can read your codebase, not just the one in your IDE.

```mermaid
graph TD
    IDE[IDE / Cursor / Claude Code] -->|MCP tool call| Server[MCP Server]
    Server -->|invoke| Graph[LangGraph StateGraph]

    Graph --> Router{Router Node}
    Router -->|needs research| Research[Research Node<br/>Gemini Pro]
    Router -->|needs design| Architect[Architect Node<br/>Claude Sonnet/Opus]
    Router -->|ready to code| Implement[Implement Node<br/>Codex]

    Research -->|findings added to state| Architect
    Architect -->|spec added to state| Implement
    Implement -->|result| Server

    subgraph "Filesystem Tools (available to all nodes)"
        Tools[read_file · glob · grep · list_dir]
    end

    Research -.->|reads codebase| Tools
    Architect -.->|reads codebase| Tools
    Implement -.->|reads & writes code| Tools

    style Server fill:#1a1a2e,stroke:#e94560,color:#fff
    style Graph fill:#0d1117,stroke:#58a6ff,color:#fff
    style Router fill:#161b22,stroke:#8b949e,color:#fff
    style Research fill:#1a73e8,stroke:#fff,color:#fff
    style Architect fill:#c45a2c,stroke:#fff,color:#fff
    style Implement fill:#2d6a4f,stroke:#fff,color:#fff
    style Tools fill:#21262d,stroke:#8b949e,color:#c9d1d9
```

### Why LangGraph?

The previous version was a simple API router — each model got a prompt string and returned text. The problem: **models had no codebase access**. You had to manually paste file contents into the `context` parameter.

With LangGraph, each node is a ReAct agent with filesystem tools. The research model can `grep` for patterns, the architect can `read_file` to understand existing code, and the implement node can read and write files directly. State flows between nodes, so research findings automatically feed into architecture decisions.

## How it works

### State

All nodes share a single `OrchestratorState` that accumulates context as the graph executes:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

class OrchestratorState(TypedDict):
    """Shared state flowing through the orchestrator graph."""
    # The original task description from the user
    task: str
    # Optional user-provided context (file contents, constraints, etc.)
    context: str
    # Classification result — which pipeline to run
    classification: dict  # {tier, confidence, reasoning, pipeline}
    # Accumulated messages from each node's agent execution
    messages: Annotated[list[AnyMessage], add_messages]
    # Research findings (populated by research node)
    research_findings: str
    # Architecture plan (populated by architect node)
    architecture_plan: str
    # Implementation result (populated by implement node)
    implementation_result: str
```

### Nodes

Each node is a factory function that returns an async handler. This pattern (closure over dependencies) keeps nodes testable and configurable:

```python
def make_research_node(model, tools):
    """Create the research node — Gemini Pro with filesystem tools."""
    agent = create_react_agent(model, tools, prompt=RESEARCH_SYSTEM_PROMPT)

    async def research_node(state: OrchestratorState) -> dict:
        result = await agent.ainvoke({"messages": [HumanMessage(content=state["task"])]})
        findings = result["messages"][-1].content
        return {
            "research_findings": findings,
            "messages": result["messages"],
        }

    return research_node
```

The same pattern applies to the architect and implement nodes — each gets a model and tools, returns a state update dict.

### Routing

A lightweight router function reads the classification and decides which nodes to visit:

```python
def route_after_classify(state: OrchestratorState) -> Literal["research", "architect", "implement"]:
    """Conditional edge — pick the next node based on classification."""
    pipeline = state["classification"].get("pipeline", ["architect"])
    if "research" in pipeline:
        return "research"
    elif "architect" in pipeline:
        return "architect"
    return "implement"
```

### Graph construction

```python
from langgraph.graph import StateGraph, START, END

def build_orchestrator_graph(config):
    """Build the full orchestrator graph."""
    # Initialize models and tools
    tools = [read_file, glob_files, grep_content, list_dir]

    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify", make_classify_node(haiku_model))
    graph.add_node("research", make_research_node(gemini_model, tools))
    graph.add_node("architect", make_architect_node(claude_model, tools))
    graph.add_node("implement", make_implement_node(codex_model, tools))

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("research", "architect")
    graph.add_edge("architect", "implement")
    graph.add_edge("implement", END)

    return graph.compile()
```

### Graph flow

```mermaid
stateDiagram-v2
    [*] --> classify
    classify --> research: pipeline includes research
    classify --> architect: pipeline includes architect (no research needed)
    classify --> implement: ready to code

    research --> architect: findings → state
    architect --> implement: plan → state
    implement --> [*]: result → MCP response
```

### Filesystem tools

Every node gets these tools so it can reason about your actual codebase:

| Tool | Description |
|---|---|
| `read_file(path)` | Read a file's contents |
| `glob_files(pattern, directory?)` | Find files matching a glob pattern |
| `grep_content(pattern, path?, file_type?)` | Search file contents with regex |
| `list_dir(path)` | List directory contents |
| `write_file(path, content)` | Write content to a file (implement node only) |

Tools are bound to nodes using the closure pattern:

```python
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path) as f:
        return f.read()

@tool
def glob_files(pattern: str, directory: str = ".") -> list[str]:
    """Find files matching a glob pattern."""
    from pathlib import Path
    return [str(p) for p in Path(directory).glob(pattern)]
```

## MCP integration

The MCP server is the external interface — your IDE calls MCP tools, which internally invoke the LangGraph graph:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-orchestrator")
graph = build_orchestrator_graph(config)

@mcp.tool()
async def chain(task_description: str, context: str = "") -> str:
    """Auto-route a task through the full pipeline."""
    result = await graph.ainvoke({
        "task": task_description,
        "context": context,
        "messages": [],
    })
    return format_result(result)
```

The individual `research()`, `architect()`, and `classify()` tools still exist for when you want to call a specific model directly.

## Setup

### 1. Install

```bash
git clone <this-repo>
cd ai-orchestrator
uv sync
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_AI_API_KEY=AIza...
#   OPENAI_API_KEY=sk-...        (for Codex implement node)
```

### 3. Connect to Cursor

Add to `.cursor/mcp.json` (project-level) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-orchestrator", "run", "ai-orchestrator"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "GOOGLE_AI_API_KEY": "AIza...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### 4. Connect to Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "ai-orchestrator": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-orchestrator", "run", "ai-orchestrator"]
    }
  }
}
```

## Tools

### `research(question, context?)`

Deep research using Gemini with filesystem access. The model can read your codebase to ground its research.

```
research("How do confidence-gated HITL pipelines work in LangGraph?")
```

### `architect(goal, context?, constraints?)`

Design an implementation plan using Claude. Reads relevant files to understand existing patterns before planning.

```
architect(
  "Add real-time WebSocket notifications for task status changes",
  constraints="Must work with existing FastAPI backend and RTK Query frontend"
)
```

### `classify(task_description)`

Fast classification — tells you which tier a task falls into and the recommended pipeline.

```
classify("Fix the typo in the dashboard header")
# → Tier: implement (confidence: 95%)
# → Pipeline: implement

classify("Add vector search to the dashboard AI chat")
# → Tier: architect (confidence: 88%)
# → Pipeline: architect → implement

classify("How does the fabrication pipeline work end to end?")
# → Tier: research (confidence: 92%)
# → Pipeline: research → architect → implement
```

### `chain(task_description, context?)`

Auto-routes through the full LangGraph pipeline. Classifies, researches if needed, architects if needed, implements if needed. Each step has full filesystem access.

```
chain("Implement a retry mechanism for failed source extractions")
# 1. Classifies → architect
# 2. Skips research (domain is understood)
# 3. Architect reads existing extraction code, designs retry strategy
# 4. Returns implementation plan with file-specific changes
```

## Configuration

Edit `config.yaml` to change models, providers, or add new roles:

```yaml
roles:
  research:
    provider: google
    model: gemini-2.0-pro          # swap models here
    max_tokens: 8192
  architect:
    provider: anthropic
    model: claude-sonnet-4-20250514  # or claude-opus-4-6
    max_tokens: 4096
  classify:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    max_tokens: 256
  implement:
    provider: openai
    model: codex-mini               # or gpt-4o
    max_tokens: 8192
```

## Project structure

```
ai-orchestrator/
├── src/orchestrator/
│   ├── __init__.py            # Package marker
│   ├── server.py              # MCP server — exposes tools, chain() invokes graph
│   ├── router.py              # Routes roles → providers (used by direct tools)
│   ├── config.py              # Loads config.yaml
│   ├── models.py              # LangChain model factories from config
│   ├── state.py               # OrchestratorState TypedDict
│   ├── graph.py               # StateGraph construction and compilation
│   ├── nodes/
│   │   ├── __init__.py        # Exports build_*_node() factories
│   │   ├── classify.py        # Classifier node (Haiku — fast routing)
│   │   ├── research.py        # Research node (Gemini — ReAct agent)
│   │   ├── architect.py       # Architect node (Claude — ReAct agent)
│   │   └── implement.py       # Implement node (placeholder for v0.3)
│   ├── tools/
│   │   ├── __init__.py        # Exports READ_TOOLS, WRITE_TOOLS
│   │   └── filesystem.py      # read_file, glob, grep, list_dir, write_file
│   ├── providers/
│   │   ├── __init__.py        # Exports Provider classes
│   │   ├── base.py            # Provider interface (used by direct tools)
│   │   ├── anthropic_provider.py
│   │   └── google_provider.py
│   └── prompts/
│       ├── __init__.py        # Exports system prompts
│       ├── research.py        # Gemini system prompt
│       ├── architect.py       # Claude system prompt
│       └── classifier.py      # Haiku classifier prompt
├── config.yaml                # Model and role config
├── pyproject.toml             # uv project config
└── .env.example               # API key template
```

## Roadmap

### v0.2 — LangGraph migration (done)

- [x] Add `langgraph`, `langchain-core`, `langchain-anthropic`, `langchain-google-genai` dependencies
- [x] Create `state.py` with `OrchestratorState` TypedDict (`add_messages` reducer)
- [x] Create `tools/filesystem.py` with `read_file`, `glob_files`, `grep_content`, `list_dir`, `write_file`
- [x] Create `models.py` — LangChain model factories reading from `OrchestratorConfig`
- [x] Create node factories in `nodes/` (classify, research, architect, implement placeholder)
- [x] Create `graph.py` with `build_orchestrator_graph()` — StateGraph with conditional routing
- [x] Update `server.py` — `chain()` invokes LangGraph graph, direct tools unchanged
- [x] Lazy graph construction — no API keys required at import time

**Two execution paths**: Direct tools (`research()`, `architect()`, `classify()`) still use the `Router` + raw providers for fast, cheap calls. The `chain()` tool uses the LangGraph graph internally with filesystem tools.

### v0.3 — Agent intelligence

Self-reflection, self-correction, checkpoints, and time-travel — the mechanics that turn a simple pipeline into a robust agent.

| Feature | How it fits | Complexity |
|---|---|---|
| **Self-reflection** | After research/architect nodes, add a "critique" node that reviews output quality and loops back if weak. Uses a cheap model (Haiku) to score the output, retries the node if below threshold. | Medium |
| **Self-correction** | Architect node validates its own plan against the actual codebase — catches hallucinated file paths, non-existent functions, wrong import paths. Runs a verification pass before returning. | Medium |
| **Checkpoints** | Add `MemorySaver` or `AsyncPostgresSaver` so multi-turn chains accumulate context across calls. Enables follow-up questions like "now add error handling to that plan". | Low |
| **Time-travel / revert** | LangGraph's built-in state history — `get_state_history(thread_id)` returns all state snapshots. Expose an MCP tool like `rewind(thread_id, checkpoint_id)` to replay from any point. Run research → architect, don't like the architecture, revert to post-research and re-run architect with different constraints. | Medium |
| **Implement node** | Wire up Codex / GPT-4o as the implement node with `write_file` tool. Takes the architect's plan and generates actual code changes. | Medium |

```mermaid
graph LR
    subgraph "v0.3 Agent Loop"
        A[Node Output] --> B{Critique<br/>Haiku}
        B -->|score < threshold| C[Retry Node]
        C --> A
        B -->|score >= threshold| D[Next Node]
    end

    subgraph "v0.3 Time Travel"
        E[checkpoint 1] --> F[checkpoint 2] --> G[checkpoint 3]
        G -.->|rewind| F
        F --> H[checkpoint 3b<br/>different constraints]
    end
```

### v0.4 — Production features

- [ ] Cost tracking — log token usage per node per request
- [ ] Streaming — `astream(stream_mode=["messages", "updates"])` for real-time output
- [ ] Custom roles — define new nodes in `config.yaml` with custom prompts
- [ ] Streamable HTTP transport — for remote hosting beyond stdio
- [ ] Rate limiting and circuit breakers per provider
