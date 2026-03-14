# LangGraph Orchestrator — Study Guide

This guide explains every design decision, data structure, and code path in the LangGraph orchestrator (Option B) in enough detail that you could reconstruct the entire codebase from scratch.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Core Concepts You Need to Know](#2-core-concepts-you-need-to-know)
3. [The State — How Data Flows](#3-the-state--how-data-flows)
4. [The Graph — How Nodes Are Wired](#4-the-graph--how-nodes-are-wired)
5. [The Supervisor — The Brain](#5-the-supervisor--the-brain)
6. [Domain Nodes — The Workers](#6-domain-nodes--the-workers)
7. [The Validator — Quality Gate](#7-the-validator--quality-gate)
8. [Fan-Out / Fan-In — Parallel Research](#8-fan-out--fan-in--parallel-research)
9. [Human-in-the-Loop — Approval Gate](#9-human-in-the-loop--approval-gate)
10. [The MCP Server — External Interface](#10-the-mcp-server--external-interface)
11. [Checkpoints and Time-Travel](#11-checkpoints-and-time-travel)
12. [Configuration and Models](#12-configuration-and-models)
13. [Output Versioning — Timeline Comparison](#13-output-versioning--timeline-comparison)
14. [How Everything Connects — Full Trace](#14-how-everything-connects--full-trace)
15. [Key Design Decisions and Why](#15-key-design-decisions-and-why)

---

## 1. What Problem Does This Solve?

You have multiple AI models (Claude, Gemini, Haiku), each good at different things. You want a system that:

- **Automatically decides** which model to use for what part of a task
- **Chains them together** — research feeds into architecture, architecture feeds into implementation
- **Self-corrects** — if output quality is low, retry with feedback
- **Remembers state** — continue a previous chain, rewind to a checkpoint
- **Parallelizes** — research multiple topics simultaneously when appropriate

The orchestrator is exposed as an **MCP server** (Model Context Protocol), meaning any IDE that speaks MCP (Cursor, Claude Code) can call it as a tool.

---

## 2. Core Concepts You Need to Know

### LangGraph StateGraph

LangGraph is a framework for building stateful, multi-step AI workflows as directed graphs. The key abstractions:

- **StateGraph**: A graph where nodes are functions and edges define the execution order. The graph is parameterized by a state type (a TypedDict).
- **Node**: A Python function (sync or async) that receives the current state and returns a dict of state updates. Only the keys you return get updated — other state keys are untouched.
- **Edge**: Connects nodes. Can be unconditional (`add_edge`) or conditional (`add_conditional_edges` — a function inspects state and returns the next node name).
- **Checkpointer**: Saves the state after every node execution. Enables time-travel (rewinding to a previous state) and multi-turn conversations (continuing from where you left off).

### How State Updates Work

This is critical to understand. When a node returns `{"research_findings": "...", "history": [...]}`:

1. LangGraph looks at each key in the returned dict
2. For keys with a **reducer** (like `messages` with `add_messages`, or `output_versions` with `operator.add`), it applies the reducer: `new_value = reducer(existing_value, returned_value)`
3. For keys **without a reducer**, it's last-writer-wins: `new_value = returned_value`

This means:
- `research_findings: str` → each write **replaces** the previous value
- `output_versions: Annotated[list[dict], operator.add]` → each write **appends** to the list
- `messages: Annotated[list[AnyMessage], add_messages]` → each write **appends** messages (with deduplication by ID)

### MCP (Model Context Protocol)

MCP is a protocol that lets IDEs discover and call tools hosted by external servers. The server runs as a subprocess communicating over stdio (stdin/stdout). The IDE sends JSON-RPC requests ("call tool X with args Y") and gets JSON-RPC responses back.

FastMCP is a Python library that handles the protocol boilerplate. You decorate functions with `@mcp.tool()` and they become available as tools. The server runs with `mcp.run()`.

### CLI Subprocesses vs API Calls

The orchestrator uses two different execution strategies:

**CLI subprocesses** (`run_claude`, `run_gemini`): Shell out to `claude -p "prompt"` or `npx @google/gemini-cli "prompt"`. The CLI tools run from the project root, so they have native codebase access — they can read files, search code, and understand project structure without being given explicit tool definitions. Used for domain work (research, architecture, implementation).

**API calls** (LangChain `ChatAnthropic`, `ChatGoogleGenerativeAI`): Direct HTTP calls to the model API. Fast and cheap, but the model only sees what you put in the prompt — no codebase access. Used for lightweight decisions (supervisor routing, validation scoring).

---

## 3. The State — How Data Flows

**File:** `src/orchestrator/state.py`

The `OrchestratorState` is a TypedDict with `total=False` (all fields optional). This is the single data structure that flows through the entire graph. Every node reads from it and writes to it.

### Field Groups

**Input fields** — set when the graph is first invoked:
```python
task: str       # "Add caching to the API" — what the user wants
context: str    # Optional additional context the user provides
```

**Domain output fields** — set by the worker nodes:
```python
research_findings: str        # Written by research node. Markdown.
architecture_plan: str        # Written by architect node. Markdown.
implementation_result: str    # Written by implement node. Text/markdown.
```

These are `str` (last-writer-wins). If the research node runs twice, only the second result survives in `research_findings`. This is intentional — consumers (architect, supervisor, validator) always want the latest version.

**Output versioning** — append-only history:
```python
output_versions: Annotated[list[dict[str, Any]], operator.add]
```

Every domain node appends a version entry here: `{"node": "research", "attempt": 2, "topic": "caching", "content": "..."}`. Because of the `operator.add` reducer, these accumulate — nothing is ever lost. This is how you compare Timeline A vs Timeline B after a rewind.

**Supervisor fields** — set by the supervisor node:
```python
next_node: str               # "research", "architect", "implement", "validator", or "finish"
supervisor_rationale: str    # "Task requires domain exploration before planning"
supervisor_instructions: str # "Research Redis vs Memcached caching patterns"
history: list[str]           # Audit trail: ["supervisor → research: ...", "research: completed ..."]
node_calls: dict[str, int]  # {"research": 2, "architect": 1} — how many times each node ran
```

`history` is a plain list (no reducer) — each node that updates it must first copy the existing list and append: `history = list(state.get("history", [])) + ["new entry"]`. This is a common LangGraph pattern for list fields without reducers.

`node_calls` is the same — copy-and-update: `node_calls = dict(state.get("node_calls", {}))` then `node_calls["research"] = node_calls.get("research", 0) + 1`.

**Fan-out fields** — for parallel research:
```python
parallel_tasks: list[dict[str, str]]  # [{"topic": "redis", "instructions": "..."}, ...]
parallel_task_topic: str              # "redis" — which sub-task this branch is handling
```

`parallel_tasks` is set by the supervisor when it wants fan-out. After the merge node runs, it's cleared to `[]`. `parallel_task_topic` is set in the `Send()` payload so each parallel branch knows its assignment.

**Validation fields** — set by the validator node:
```python
validation_score: float      # 0.0 to 1.0
validation_feedback: str     # "Missing cache invalidation strategy"
```

**Chat history** (legacy, carried over from earlier versions):
```python
messages: Annotated[list[AnyMessage], add_messages]
```

Uses LangGraph's `add_messages` reducer which handles message deduplication by ID. Not actively used by the v0.5 nodes (they use CLI subprocesses, not chat-based agents), but kept for backward compatibility and potential future use.

### Why `total=False`?

Without `total=False`, TypedDict requires every field to be present in every state dict. Since different nodes write different fields, and the initial state only has `task` (and maybe `context`), we need all fields to be optional. Nodes use `.get("field", default)` to safely read optional fields.

---

## 4. The Graph — How Nodes Are Wired

**File:** `src/orchestrator/graph.py`

### The Build Function

`build_orchestrator_graph(config)` is the factory that assembles and compiles the graph:

1. Creates the supervisor model (Haiku via `get_classify_model(config)`)
2. Builds each node using factory functions
3. Registers nodes on a `StateGraph(OrchestratorState)`
4. Wires edges (unconditional and conditional)
5. Compiles with the `InMemorySaver` checkpointer

The compiled graph is a callable — you invoke it with `await graph.ainvoke(state, config={"configurable": {"thread_id": "..."}})`.

### Node Registration

```python
graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_node)
graph.add_node("architect", architect_node)
graph.add_node("implement", implement_node)
graph.add_node("validator", validator_node)
graph.add_node("merge_research", _merge_research_node)
```

Each name is a string key. Edge definitions reference these keys. The node value is an async function `(state) -> dict`.

### Edge Wiring — The Execution Flow

**Entry edge:**
```python
graph.add_edge(START, "supervisor")
```
Every graph execution starts at the supervisor. The supervisor inspects the (possibly empty) state and decides what to do first.

**Supervisor conditional edges:**
```python
graph.add_conditional_edges(
    "supervisor",
    select_next_node,          # routing function
    {                          # path map
        "research": "research",
        "architect": "architect",
        "implement": "implement",
        "validator": "validator",
        END: END,
    },
)
```

After the supervisor runs, LangGraph calls `select_next_node(state)`. This function can return:
- A string like `"research"` → routed via the path map to the `research` node
- The special `END` constant → graph terminates
- A list of `Send()` objects → fan-out (bypasses the path map entirely)

**Research conditional exit:**
```python
graph.add_conditional_edges(
    "research",
    _research_exit,
    {
        "merge_research": "merge_research",
        "supervisor": "supervisor",
    },
)
```

After the research node runs, `_research_exit(state)` checks if `parallel_tasks` is populated:
- Yes → route to `merge_research` (fan-in)
- No → route back to `supervisor` (sequential)

**Fixed return edges:**
```python
graph.add_edge("merge_research", "supervisor")
graph.add_edge("architect", "supervisor")
graph.add_edge("implement", "supervisor")
graph.add_edge("validator", "supervisor")
```

Every non-research node always returns to the supervisor. This is the hub-and-spoke pattern — the supervisor is the hub, domain nodes are spokes.

### The `select_next_node` Function — Routing Logic

```python
def select_next_node(state: OrchestratorState) -> str | list[Send]:
    next_node = state.get("next_node", "finish")
    if next_node == "finish":
        return END

    parallel_tasks = state.get("parallel_tasks", [])

    if next_node == "research" and parallel_tasks:
        return [
            Send("research", {
                "task": state.get("task", ""),
                "context": state.get("context", ""),
                "supervisor_instructions": pt.get("instructions", ""),
                "parallel_task_topic": pt.get("topic", ""),
                "validation_feedback": state.get("validation_feedback", ""),
            })
            for pt in parallel_tasks
        ]

    return next_node
```

Three code paths:
1. `next_node == "finish"` → return `END` → graph terminates
2. `next_node == "research"` AND `parallel_tasks` is non-empty → return list of `Send()` → fan-out
3. Everything else → return the node name string → sequential routing

Each `Send("research", payload)` dispatches the `research` node with a custom input dict (not the full state). The research node's `.get()` calls handle both full state and partial payload gracefully because of `total=False`.

### The Checkpointer

```python
checkpointer = InMemorySaver()
```

Module-level singleton. Every `graph.compile(checkpointer=checkpointer)` call uses the same checkpointer, so all threads share one persistence store.

`InMemorySaver` stores checkpoints in a Python dict in memory. State survives across multiple `ainvoke()` calls within the same server process, but is lost when the process restarts. For production, you'd swap this with `SqliteSaver` or `PostgresSaver`.

When you call `graph.ainvoke(state, config={"configurable": {"thread_id": "abc"}})`:
- LangGraph saves a checkpoint after every node execution
- The checkpoint contains the full state at that point
- The thread_id groups checkpoints together
- `aget_state_history(config)` returns all checkpoints for a thread
- `aget_state(config)` returns the latest (or a specific checkpoint_id)

---

## 5. The Supervisor — The Brain

**File:** `src/orchestrator/nodes/supervisor.py`

The supervisor is the most important node. It runs after every other node (and at the very start) and makes all routing decisions.

### How It Works

1. Reads the full state: task, context, history, node call counts, domain outputs (truncated), validation results
2. Formats everything into a state summary (markdown sections)
3. Sends the summary to Haiku with the system prompt
4. Haiku returns a `RouterDecision` (Pydantic structured output)
5. The decision is written to state as `next_node`, `supervisor_rationale`, `supervisor_instructions`, and `parallel_tasks`

### The System Prompt

The `SUPERVISOR_SYSTEM_PROMPT` tells Haiku:
- What each node does (research, architect, implement, validator, finish)
- Decision rules (when to use each node)
- Fan-out rules (when to use parallel research)
- What state fields to inspect

This prompt is the "brain" of the system. The quality of routing depends entirely on how well this prompt is written.

### Pydantic Structured Output

```python
structured_model = model.with_structured_output(RouterDecision)
decision: RouterDecision = await structured_model.ainvoke(messages)
```

`with_structured_output(RouterDecision)` configures the model to return a JSON object matching the Pydantic schema. LangChain handles the conversion — the model sees the schema as a tool definition, returns JSON, and LangChain validates + parses it into a `RouterDecision` instance.

This is more reliable than asking the model to return JSON and parsing it yourself:
- The schema is enforced at the model level (tool use / function calling)
- Pydantic validates the response (correct types, valid enum values)
- You get Python objects, not strings

### The RouterDecision Schema

```python
class ParallelTask(BaseModel):
    topic: str        # "redis-caching"
    instructions: str # "Research Redis caching patterns for REST APIs"

class RouterDecision(BaseModel):
    next_step: Literal["research", "architect", "implement", "validator", "finish"]
    rationale: str         # "The task involves an unknown domain that needs exploration"
    instructions: str      # "Research caching patterns for REST APIs"
    parallel_tasks: list[ParallelTask] = Field(default_factory=list)
```

`next_step` is a `Literal` type — the model can only return one of the five valid values. If it tries to return something else, Pydantic validation fails and LangChain retries.

`parallel_tasks` defaults to an empty list. The model only populates it when it decides to fan out. This makes fan-out opt-in — the sequential path is the default.

### State Summary Construction

The supervisor doesn't see the raw state dict. It sees a formatted markdown summary:

```python
state_summary_parts = [f"## Task\n\n{task}"]
if context:
    state_summary_parts.append(f"## Context\n\n{context}")
if history:
    state_summary_parts.append("## History\n\n" + "\n".join(f"- {h}" for h in history))
if node_calls:
    state_summary_parts.append(f"## Node call counts\n\n{counts}")
# ... truncated outputs, validation results
```

Domain outputs are truncated to keep the prompt short:
- Research findings: max 1000 chars
- Architecture plan: max 1000 chars
- Implementation result: max 500 chars

This is a cost/quality tradeoff. Haiku is cheap but has limited context. The supervisor doesn't need to see the full 10,000-word research output — it just needs to know research was done and what it covered.

### Why Haiku?

The supervisor model needs to be:
1. **Fast** — it runs after every node, so latency adds up
2. **Cheap** — it runs many times per chain
3. **Good at structured output** — it needs to return valid JSON matching the schema

Haiku excels at all three. It's the fastest Anthropic model and costs ~10x less than Sonnet. Structured output (which node to call next) is a simple task that doesn't need a large model.

---

## 6. Domain Nodes — The Workers

### Research Node (`nodes/research.py`)

**What it does:** Shells out to Gemini CLI for deep research.

**Input it reads from state:**
- `task` — what to research
- `context` — additional context
- `supervisor_instructions` — specific research instructions from the supervisor
- `validation_feedback` — feedback from a previous validation (for retries)
- `parallel_task_topic` — topic label when running as a parallel branch

**What it writes to state:**
- `research_findings` — the research output (markdown)
- `output_versions` — version entry with node, attempt, topic, content
- `node_calls` — incremented research count
- `history` — step summary

**How it builds the prompt:**

```python
prompt = build_prompt(
    RESEARCH_SYSTEM_PROMPT,    # System-level instructions for research behavior
    task,                       # The user's task
    context section,            # Optional
    instructions section,       # From supervisor
    feedback section,           # From validator (if retrying)
)
```

`build_prompt()` (from `cli_server_pkg/helpers/prompts.py`) concatenates non-empty sections with `\n\n---\n\n` separators.

**The `run_gemini` call:**

```python
findings = await run_gemini(prompt, timeout=600)
```

`run_gemini` (from `cli_server_pkg/session/runners.py`) spawns `npx @google/gemini-cli "prompt"` as a subprocess. The 600-second timeout kills the process if it hangs. The CLI runs from the project root, giving Gemini native access to read and search the codebase.

**Parallel mode:** When called via `Send()`, the node receives a partial dict instead of the full state. The `.get()` calls handle this — missing fields default to empty strings. The `parallel_task_topic` is prepended to instructions: `[Research sub-task: redis-caching]\n\n{instructions}`.

### Architect Node (`nodes/architect.py`)

**What it does:** Shells out to Claude Code CLI for design and planning.

**Input it reads from state:**
- `task`, `context`, `research_findings`, `supervisor_instructions`, `validation_feedback`

**What it writes to state:**
- `architecture_plan`, `output_versions`, `node_calls`, `history`

**Self-correction:** The prompt includes a special section:

```
## Self-correction

Before finalizing your plan, verify that every file path and function
name you reference actually exists in the codebase. Read the files to
confirm. If you find a hallucinated path or name, fix it.
```

This is appended to the prompt via `build_prompt()`. Claude Code reads the instruction, and because it has native codebase access, it can actually verify paths and names before finalizing. This dramatically reduces hallucinated file paths in architecture plans.

### Implement Node (`nodes/implement.py`)

**What it does:** Shells out to Claude Code CLI with full read/write access.

**Input it reads from state:**
- `task`, `context`, `architecture_plan`, `supervisor_instructions`

**What it writes to state:**
- `implementation_result`, `output_versions`, `node_calls`, `history`

**Key difference:** This node has its own system prompt (`IMPLEMENT_SYSTEM_PROMPT`) defined directly in the file, not imported from `prompts/`. The prompt is specifically about implementation: follow the plan precisely, match existing code style, don't redesign, don't leave TODOs.

**Why the implement node doesn't read `validation_feedback`:** The supervisor's decision rules say "never validate after implement." Implementation is the final domain step — the output is actual code changes, not a document to be scored. If the implementation is wrong, you'd rewind and fix the architecture, not ask the validator to score code.

### The Node Factory Pattern

All domain nodes use the same pattern:

```python
def build_research_node():          # Factory function (no params for CLI nodes)
    async def research_node(state):  # Closure — the actual node
        # Read state
        # Build prompt
        # Call CLI
        # Return state update dict
    return research_node             # Return the closure
```

This pattern exists because:
1. **Consistency** — all nodes have the same interface: `async (state) -> dict`
2. **Testability** — you can create a node and test it without the full graph
3. **Configurability** — supervisor/validator nodes take a `model` parameter via `build_supervisor_node(model)`

For CLI-based nodes (research, architect, implement), the factory takes no parameters because the CLI is discovered at runtime. For API-based nodes (supervisor, validator), the factory takes a `model` parameter so you can inject different models for testing.

---

## 7. The Validator — Quality Gate

**File:** `src/orchestrator/nodes/validator.py`

### What It Does

Scores the most recent domain output (research findings or architecture plan) on a 0.0-1.0 scale. Returns a score and actionable feedback. The supervisor reads this to decide: proceed, or retry the node with the feedback?

### How It Works

1. Determines what to validate: checks `architecture_plan` first, then `research_findings`. If neither exists, returns score 1.0 (nothing to validate, pass through).
2. Sends the output to Haiku with the validator system prompt
3. Haiku returns JSON: `{"score": 0.75, "feedback": "..."}`
4. The response is parsed (with fallback for JSON parse errors)
5. Score and feedback are written to state

### Scoring Criteria

Four criteria, each worth 0.25:

| Criterion | Question |
|-----------|----------|
| Completeness | Does it address the full scope of the task? |
| Specificity | Does it reference concrete details (file paths, function names)? |
| Actionability | Could someone act on this without asking follow-up questions? |
| Accuracy | Does it avoid hallucinations, vague hand-waving, generic advice? |

### Why JSON Instead of Structured Output?

The validator uses plain `model.ainvoke()` and parses JSON manually, unlike the supervisor which uses `with_structured_output()`. This is because:
- The validator schema is simple (just score + feedback)
- JSON parsing is sufficient — no need for Pydantic validation
- The fallback (score 0.5 on parse error) is acceptable

If the model wraps the JSON in markdown code fences (```json ... ```), the parser strips them.

### How the Supervisor Uses Validation

The supervisor's system prompt says:
- "After research or architect: validate the output quality"
- "If the validator reports low quality: retry the node with the feedback"
- "After implement: finish (do not validate implementation)"

In practice, a typical flow is:
1. Supervisor → research
2. Supervisor → validator (score: 0.65, feedback: "Missing X")
3. Supervisor → research (with feedback in `validation_feedback`)
4. Supervisor → validator (score: 0.85, no feedback)
5. Supervisor → architect
6. ...

The validator never runs after implement — that's a hard rule in the supervisor prompt. Implementation quality is measured by whether the code works, not by a text scorer.

---

## 8. Fan-Out / Fan-In — Parallel Research

### The Problem

In sequential mode, researching three topics means:
```
supervisor → research(redis) → supervisor → research(memcached) → supervisor → research(cdn) → supervisor
```

Six supervisor calls, three sequential research calls. Slow.

### The Solution: Send()

LangGraph's `Send()` API dispatches multiple instances of the same node in parallel:

```
supervisor → Send([research(redis), research(memcached), research(cdn)]) → merge_research → supervisor
```

One supervisor call, three concurrent research calls, one merge. The total time is roughly `max(research_time_1, research_time_2, research_time_3)` instead of `sum(...)`.

### How Send() Works Under the Hood

When `select_next_node` returns a list of `Send()` objects:

1. LangGraph ignores the path map (the `{...}` dict in `add_conditional_edges`)
2. For each `Send("research", payload)`, it creates an independent execution of the `research` node
3. Each execution receives `payload` as its state argument (not the full graph state)
4. All executions run concurrently (asyncio tasks)
5. Each execution returns a state update dict
6. LangGraph applies all updates to the graph state using reducers:
   - `output_versions` uses `operator.add` → all branches' entries are concatenated
   - `research_findings` is last-writer-wins → one branch "wins" (arbitrary)
   - `history` is last-writer-wins → one branch's history entry survives

This is why `merge_research` exists — to properly combine the parallel results that `last-writer-wins` would discard.

### The Send() Payload

```python
Send("research", {
    "task": state.get("task", ""),
    "context": state.get("context", ""),
    "supervisor_instructions": pt.get("instructions", ""),
    "parallel_task_topic": pt.get("topic", ""),
    "validation_feedback": state.get("validation_feedback", ""),
})
```

This is a plain dict, not an `OrchestratorState`. The research node's `.get()` calls handle both cases transparently. Fields not in the payload (like `node_calls`, `history`) return their defaults (`{}`, `[]`).

### The Merge Node

```python
async def _merge_research_node(state: OrchestratorState) -> dict:
```

After all parallel branches complete:
1. Reads `output_versions` — finds entries where `node == "research"` and `topic` matches one of the `parallel_tasks` topics
2. Combines them into sectioned markdown: `### topic\n\ncontent` separated by `---`
3. Writes the combined result to `research_findings` (overwriting the arbitrary "winner")
4. Clears `parallel_tasks` to `[]` — prevents `_research_exit` from routing to merge on subsequent sequential research calls

### The Conditional Exit

```python
def _research_exit(state: OrchestratorState) -> str:
    if state.get("parallel_tasks", []):
        return "merge_research"
    return "supervisor"
```

This is critical: it runs after **every** research node execution, including each parallel branch. During fan-out, `parallel_tasks` is still populated (the supervisor set it, merge hasn't cleared it yet), so all branches route to `merge_research`. LangGraph waits for all branches to complete before running `merge_research`.

In sequential mode, `parallel_tasks` is empty (either never set, or cleared by a previous merge), so research routes directly back to supervisor.

### When Does the Supervisor Fan Out?

The system prompt gives guidelines:
- 2-4 independent knowledge gaps
- Evaluating competing technologies
- Researching different aspects of a system
- Comparing frameworks or libraries

The supervisor decides this autonomously based on the task. A task like "Evaluate Redis vs Memcached vs CDN caching" is a strong candidate. A task like "How does Redis pub/sub work?" is not — it's a single focused question.

---

## 9. Human-in-the-Loop — Approval Gate

**File:** `src/orchestrator/nodes/human_review.py`

### The Problem

The orchestrator can research, design, and implement code changes autonomously. But you don't want it implementing code without your approval — a bad architecture decision could waste hours or break production code. You need a pause point where you review the plan before any code is written.

### How interrupt() Works

LangGraph's `interrupt()` function is the primitive for pausing a graph:

```python
from langgraph.types import interrupt

async def human_review_node(state):
    plan = state.get("architecture_plan", "")

    # First execution: raises GraphInterrupt, pausing the graph
    # On resume: returns the value from Command(resume=...)
    response = interrupt({
        "message": "Review this plan before implementation",
        "architecture_plan": plan,
    })

    # This code only runs AFTER resume
    decision = response.get("decision", "approved")
    feedback = response.get("feedback", "")
    return {"human_review_status": decision, "human_feedback": feedback}
```

The critical thing to understand: **the node re-executes from the top on resume**. When `interrupt()` is called the first time, it raises an exception — nothing after it runs. When the graph is resumed with `Command(resume=value)`, the entire node function runs again, but this time `interrupt()` returns the resume value instead of raising.

This means:
1. Any state reads before `interrupt()` happen twice (once on pause, once on resume)
2. The resume value is the return of `interrupt()`, not a separate callback
3. A checkpointer is required — the graph state must be saved so it can be restored

### The Resume Mechanism

The caller resumes the graph with `Command(resume=...)`:

```python
from langgraph.types import Command

# Resume with approval
result = await graph.ainvoke(
    Command(resume={"decision": "approved", "feedback": ""}),
    config={"configurable": {"thread_id": thread_id}},
)

# Resume with rejection
result = await graph.ainvoke(
    Command(resume={"decision": "rejected", "feedback": "Add error handling to step 3"}),
    config={"configurable": {"thread_id": thread_id}},
)
```

`Command` is a LangGraph primitive that tells the graph "continue from where you paused, and give this value to the `interrupt()` call." The graph loads the checkpoint from the thread_id, re-enters the `human_review` node, and `interrupt()` returns the resume value.

### The MCP Integration

Two MCP tools handle the HITL flow:

**`chain()`** detects when the graph pauses:
```python
result = await graph.ainvoke(initial_state, config=graph_config)

# Check if paused at human_review
state = await graph.aget_state(graph_config)
if state and state.next and "human_review" in state.next:
    # Graph is paused — return the plan with approval instructions
    return f"Waiting for approval... call approve(thread_id='{thread_id}')"
```

After `ainvoke()`, the graph state shows `next = ["human_review"]` — meaning the `human_review` node was about to run (or is paused mid-execution). The `chain()` tool checks this and returns the plan with instructions for the user.

**`approve(thread_id, feedback?)`** resumes the graph:
```python
resume_value = {"decision": "approved" if not feedback else "rejected", "feedback": feedback}
result = await graph.ainvoke(Command(resume=resume_value), config=graph_config)
```

### What Happens After Resume

The supervisor sees the review result in state:
- `human_review_status: "approved"` → route to implement
- `human_review_status: "rejected"` + `human_feedback: "Add error handling"` → route to architect with the feedback

If the architect revises and the plan goes through validation again, the supervisor routes to `human_review` again — you get another chance to review the revised plan.

### State Fields

```python
human_review_status: str    # "approved" or "rejected"
human_feedback: str          # Empty if approved, feedback text if rejected
```

Both are last-writer-wins. The supervisor reads them to decide what to do after the review.

### Why Not Just Check Before implement?

You could skip the HITL node and have the `chain()` tool check if the supervisor chose "implement" and pause there. But that breaks the graph abstraction — the MCP tool would need to understand the graph's internal routing logic. The HITL node keeps the pause/resume logic inside the graph where it belongs. The supervisor explicitly routes to `human_review`, and the graph handles the rest.

---

## 10. The MCP Server — External Interface

**File:** `src/orchestrator/server.py`

### Server Setup

```python
load_dotenv()  # Load .env before any imports that read env vars
# ... imports ...
config = load_config()
router = Router(config)
mcp = FastMCP("ai-orchestrator")
```

Key: `load_dotenv()` runs before the config/router imports so API keys from `.env` are available as environment variables.

The graph is built lazily:
```python
_orchestrator_graph = None

def _get_graph():
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph(config)
    return _orchestrator_graph
```

This avoids building the graph (and validating API keys) at import time. The graph is only built on the first `chain()`, `history()`, or `rewind()` call.

### Tool Categories

**Direct tools** — bypass the graph entirely:
- `research(question, context?)` → calls `run_gemini()` directly
- `architect(goal, context?, constraints?)` → calls `run_claude()` directly
- `classify(task_description)` → uses the `Router` to call Haiku API

These are fast, single-model calls for when you know exactly which model you need.

**Graph tools** — invoke the full LangGraph pipeline:
- `chain(task_description, context?, thread_id?)` → builds state, invokes graph, pauses for approval before implementation
- `approve(thread_id, feedback?)` → resumes a paused chain with approval or rejection
- `history(thread_id, limit?)` → reads checkpoints from the checkpointer
- `rewind(thread_id, checkpoint_id, new_task?)` → loads a checkpoint, re-invokes graph

### The `chain()` Tool — Full Pipeline with Streaming

```python
@mcp.tool()
async def chain(task_description: str, ctx: Context, context: str = "", thread_id: str = "") -> str:
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph_config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"task": task_description}
    if context:
        initial_state["context"] = context

    graph = _get_graph()

    # Stream updates — get real-time progress
    result = {}
    step = 0
    async for update in graph.astream(initial_state, config=graph_config, stream_mode="updates"):
        step += 1
        for node_name, state_update in update.items():
            result.update(state_update)
            message = _build_progress_message(node_name, state_update)
            await ctx.report_progress(step, message=message)

    # ... check for HITL pause, format result ...
```

Step by step:
1. Generate a thread_id (or use the one provided for multi-turn)
2. Create the initial state dict
3. Get (or build) the graph
4. **Stream** the graph via `astream(stream_mode="updates")` instead of `ainvoke()`
5. After each node completes, `_build_progress_message()` formats a short update
6. `ctx.report_progress()` sends an MCP progress notification to the IDE
7. If the graph pauses at `human_review`, return the plan with approval instructions
8. Otherwise, format the final result and return

### Streaming Progress — How It Works

`graph.astream(stream_mode="updates")` yields a dict after each node completes. The dict is `{node_name: state_update}` — the same state update the node returned.

`_build_progress_message()` extracts the relevant info from each node's update:
- **Supervisor:** shows the routing decision, rationale, and fan-out topics
- **Validator:** shows the quality score and feedback
- **Research/Architect/Implement:** shows completion status and topic labels
- **Human Review:** shows the approval/rejection status

`ctx.report_progress(step, message=message)` sends an MCP `notifications/progress` message to the IDE. FastMCP injects the `Context` object automatically when the function declares a `ctx: Context` parameter.

The key difference from `ainvoke()`: with streaming, the IDE sees progress **while the pipeline runs** — each supervisor decision, each validation score, each node completion appears in real time. Without streaming, you wait for the entire pipeline to finish before seeing anything.

The `thread_id` is returned in the response so the caller can continue the chain later.

### Result Formatting

`_format_graph_result(state)` builds a markdown response from the final state:

1. **Supervisor Journey** — the full `history` list, numbered
2. **Quality** — validation score and feedback (if present)
3. **Research Findings** — full research output
4. **Architecture Plan** — full plan output
5. **Implementation** — full implementation output

Sections are separated by `---`. Only sections with content are included.

### The `history()` Tool — Checkpoint Inspection

Iterates over `graph.aget_state_history(config)` and formats each checkpoint:
- Step number and checkpoint ID (truncated)
- Source (which node created this checkpoint)
- What data exists (research_findings, architecture_plan, implementation_result)
- Supervisor decision (next_node and rationale)
- Validation score
- Output version counts (how many versions per node)

### The `rewind()` Tool — Time Travel

```python
snapshot = await graph.aget_state(config)  # Load checkpoint
result = await graph.ainvoke(input_state or None, config=config)  # Re-run from there
```

By including `checkpoint_id` in the config, `aget_state` loads that specific checkpoint. The subsequent `ainvoke` resumes execution from the next node in the checkpoint's `next` list. If `new_task` is provided, it overrides the task in the state.

---

## 11. Checkpoints and Time-Travel

### What Gets Checkpointed

After every node execution, `InMemorySaver` saves:
- The full state at that point
- Metadata: step number, source node, timestamp
- The `next` list: which node(s) would execute next

### Thread Model

Each `chain()` call creates (or continues) a thread identified by `thread_id`. A thread is a sequence of checkpoints:

```
thread abc-123:
  step 0: START → supervisor           (state: {task: "..."})
  step 1: supervisor → research        (state: {task, next_node: "research", ...})
  step 2: research → supervisor        (state: {task, research_findings: "...", ...})
  step 3: supervisor → validator       (state: {task, research_findings, next_node: "validator", ...})
  ...
```

### Multi-Turn Chains

If you call `chain("Add caching", thread_id="abc")` and then later call `chain("Now add tests", thread_id="abc")`, the second call:
1. Loads the latest checkpoint for thread "abc"
2. The existing state (research findings, architecture plan, etc.) is preserved
3. The new task overrides `state["task"]`
4. The supervisor sees the full accumulated context and decides what to do next

### Rewinding

`rewind(thread_id, checkpoint_id)`:
1. Loads the state at that specific checkpoint
2. Resumes execution from whatever node was `next` at that checkpoint
3. The graph runs forward from there, creating new checkpoints
4. The old checkpoints are preserved — you now have a branching timeline

This is why `output_versions` uses an append reducer — when you rewind and re-run, both the original and new versions accumulate, letting you compare timelines.

---

## 12. Configuration and Models

### Config Loading (`config.py`)

```python
def load_config(config_path=None) -> OrchestratorConfig:
```

Search order:
1. Explicit path argument
2. `./config.yaml` (project root)
3. `~/.config/ai-orchestrator/config.yaml`
4. Hardcoded defaults

The default config defines three roles:
- `research`: Google / `gemini-2.0-pro`
- `architect`: Anthropic / `claude-sonnet-4`
- `classify`: Anthropic / `claude-haiku-4-5` (max 256 tokens)

### Model Factories (`models.py`)

```python
def get_classify_model(config) -> BaseChatModel:
    return _build_model(config, "classify")
```

`_build_model` reads the role config, resolves the API key from environment, and creates the appropriate LangChain model:
- `provider == "anthropic"` → `ChatAnthropic(model=..., api_key=...)`
- `provider == "google"` → `ChatGoogleGenerativeAI(model=..., google_api_key=...)`

API keys are resolved from environment variables (names configured in `ProviderConfig.api_key_env`). If the env var isn't set, the key is omitted and the SDK falls back to its own env var resolution.

### Who Uses What

| Component | Config Role | Model Type | How Called |
|-----------|------------|------------|-----------|
| Supervisor | classify | Haiku (API) | `model.with_structured_output(RouterDecision)` |
| Validator | classify | Haiku (API) | `model.ainvoke(messages)` |
| Research | — | Gemini (CLI) | `run_gemini(prompt)` subprocess |
| Architect | — | Claude (CLI) | `run_claude(prompt)` subprocess |
| Implement | — | Claude (CLI) | `run_claude(prompt)` subprocess |

The supervisor and validator share the same model instance (Haiku). Domain nodes don't use models at all — they spawn CLI subprocesses.

---

## 13. Output Versioning — Timeline Comparison

### The Problem

With `research_findings: str` (last-writer-wins), when the research node runs twice (e.g., once before validation feedback, once after), the first result is gone. When you rewind and re-run, the original timeline's output is overwritten.

### The Solution

`output_versions: Annotated[list[dict[str, Any]], operator.add]` accumulates every output:

```python
# Research node returns:
{
    "research_findings": findings,     # Latest (overwrites)
    "output_versions": [{              # History (appends)
        "node": "research",
        "attempt": 2,
        "topic": "caching",
        "content": findings,
    }],
}
```

After two research runs and one architect run, `output_versions` looks like:
```python
[
    {"node": "research", "attempt": 1, "topic": "sequential", "content": "...first research..."},
    {"node": "research", "attempt": 2, "topic": "sequential", "content": "...second research..."},
    {"node": "architect", "attempt": 1, "topic": "sequential", "content": "...plan..."},
]
```

After a rewind and re-run of research:
```python
[
    # ... original entries ...
    {"node": "research", "attempt": 1, "topic": "sequential", "content": "...timeline B research..."},
]
```

Both timelines' outputs are preserved. The `history()` MCP tool shows version counts at each checkpoint so you can see where timelines diverge.

---

## 14. How Everything Connects — Full Trace

Let's trace `chain("Evaluate Redis vs Memcached for our API caching layer")` from start to finish:

### Step 0: MCP Tool Call

1. IDE sends JSON-RPC: `{"method": "tools/call", "params": {"name": "chain", "arguments": {"task_description": "Evaluate Redis..."}}}`
2. FastMCP dispatches to the `chain()` function
3. `chain()` generates `thread_id = "a1b2c3..."`, builds `initial_state = {"task": "Evaluate Redis..."}`
4. Calls `graph.ainvoke(initial_state, config={"configurable": {"thread_id": "a1b2c3"}})`

### Step 1: Supervisor (first call)

1. `supervisor_node(state)` runs. State has only `task`.
2. Builds state summary: just `## Task\n\nEvaluate Redis vs Memcached...`
3. Sends to Haiku with `SUPERVISOR_SYSTEM_PROMPT`
4. Haiku returns:
   ```json
   {
     "next_step": "research",
     "rationale": "Need to understand both technologies before making a recommendation",
     "instructions": "Compare Redis and Memcached for REST API caching",
     "parallel_tasks": [
       {"topic": "redis-caching", "instructions": "Research Redis caching patterns, pros/cons for REST APIs"},
       {"topic": "memcached-caching", "instructions": "Research Memcached caching patterns, pros/cons for REST APIs"}
     ]
   }
   ```
5. Returns state update: `{next_node: "research", parallel_tasks: [...], history: ["supervisor → research: ..."], ...}`
6. Checkpoint saved (step 1)

### Step 2: Fan-Out

1. `select_next_node(state)` runs
2. Sees `next_node == "research"` AND `parallel_tasks` is non-empty
3. Returns `[Send("research", {task, parallel_task_topic: "redis-caching", ...}), Send("research", {task, parallel_task_topic: "memcached-caching", ...})]`

### Step 3: Parallel Research (concurrent)

**Branch A (redis-caching):**
1. `research_node({"task": "Evaluate Redis...", "parallel_task_topic": "redis-caching", "supervisor_instructions": "Research Redis caching..."})`
2. Builds prompt with `[Research sub-task: redis-caching]` prepended
3. `run_gemini(prompt, timeout=600)` → spawns Gemini CLI
4. Returns `{research_findings: "Redis analysis...", output_versions: [{node: "research", topic: "redis-caching", ...}], ...}`

**Branch B (memcached-caching):** Same, but for Memcached.

Both run concurrently. LangGraph applies updates:
- `research_findings` → one branch wins (arbitrary)
- `output_versions` → both entries appended (operator.add reducer)

### Step 4: Research Exit

1. `_research_exit(state)` runs for each branch
2. `parallel_tasks` is still `[{redis}, {memcached}]` (supervisor set it, merge hasn't cleared it)
3. Returns `"merge_research"` for both branches
4. LangGraph waits for both branches, then runs `merge_research`

### Step 5: Merge Research

1. `_merge_research_node(state)` runs
2. Reads `output_versions` — finds entries with topics `redis-caching` and `memcached-caching`
3. Combines into: `### redis-caching\n\n...\n\n---\n\n### memcached-caching\n\n...`
4. Returns `{research_findings: "combined...", parallel_tasks: [], history: [..., "merge_research: combined 2 parallel findings"]}`
5. Checkpoint saved

### Step 6: Back to Supervisor

1. Supervisor sees: history (3 entries), research_findings (combined), no plan yet
2. Decides: `next_step: "validator"`, rationale: "Check quality of combined research"
3. Checkpoint saved

### Step 7: Validator

1. Sees `research_findings` (combined), no `architecture_plan`
2. Sends to Haiku: "Score this research output"
3. Haiku returns `{score: 0.88, feedback: ""}`
4. Returns `{validation_score: 0.88, validation_feedback: "", history: [..., "validator: research findings scored 0.88"]}`
5. Checkpoint saved

### Step 8: Supervisor Again

1. Sees: good research (0.88), no plan yet
2. Decides: `next_step: "architect"`, instructions: "Design caching architecture based on Redis vs Memcached findings"
3. Checkpoint saved

### Step 9: Architect

1. Reads `task`, `research_findings`, `supervisor_instructions`
2. Builds prompt with self-correction instructions
3. `run_claude(prompt, timeout=600)` → spawns Claude Code CLI
4. Claude reads the codebase, designs a plan, verifies file paths
5. Returns `{architecture_plan: "...", output_versions: [{node: "architect", ...}], ...}`
6. Checkpoint saved

### Step 10: Validate

1. Supervisor routes to validator
2. Validator scores the plan: 0.85 → passes
3. Checkpoint saved

### Step 11: Supervisor → Human Review

1. Supervisor sees: validated plan (0.85)
2. Decides: `next_step: "human_review"` — plan must be approved before implementation
3. Checkpoint saved

### Step 12: Human Review (PAUSED)

1. `human_review_node` runs, calls `interrupt(review_payload)`
2. Graph execution **stops** — `GraphInterrupt` exception raised
3. `chain()` detects the pause via `state.next == ["human_review"]`
4. Returns the plan to the IDE with: "Waiting for approval... call `approve(thread_id)`"
5. The user reviews the plan in the IDE

### Step 13: Approve (RESUMED)

1. User calls `approve(thread_id="a1b2c3")`
2. `approve()` sends `Command(resume={"decision": "approved", "feedback": ""})`
3. `human_review_node` re-executes — `interrupt()` returns the resume value
4. Node writes `{human_review_status: "approved", human_feedback: ""}` to state
5. Checkpoint saved, graph continues

### Step 14: Supervisor → Implement → Finish

1. Supervisor sees: `human_review_status: "approved"` → routes to implement
2. Implement runs via Claude Code CLI
3. Supervisor sees implementation → finishes
4. `graph.ainvoke()` returns the final state

### Step 15: Result Returned

`_format_graph_result()` builds the markdown response including the human review status. `chain()` (via `approve()`) returns it to the IDE via MCP.

---

## 15. Key Design Decisions and Why

### Why Hub-and-Spoke Instead of Linear Pipeline?

The v0.3 pipeline was linear: `classify → research → critique → architect → critique → implement`. Problems:
- Fixed order — can't skip research for simple tasks
- Can't retry a specific node without re-running everything
- Adding a node means rewiring the entire pipeline

Hub-and-spoke lets the supervisor dynamically decide the order. It can skip research, retry architect 3 times, or go straight to implement. Adding a new node just means registering it and updating the `RouterDecision` literal.

### Why CLI Subprocesses Instead of API + ReAct Agents?

v0.2 used LangChain ReAct agents with custom filesystem tools. Problems:
- The model discovers files one tool call at a time (slow, expensive)
- Custom tools can't match the built-in search/indexing of CLI tools
- Each tool call is a round-trip to the API

CLI subprocesses give the model native codebase access — built-in search, indexing, context management. A single subprocess call replaces dozens of tool calls.

### Why Separate `research_findings` (str) and `output_versions` (list)?

Consumers (architect, supervisor, validator) always want the latest output as a simple string. Making them iterate over a list to find the latest entry would add complexity to every consumer.

`output_versions` serves a different purpose: historical comparison, debugging, timeline analysis. It's read by the `history()` tool and the merge node, not by domain consumers.

### Why `parallel_tasks` Defaults to Empty (Not None)?

Pydantic's `default_factory=list` means the model doesn't need to explicitly set `parallel_tasks: []` for sequential research. This makes the common case (sequential) require zero extra work from the model, and the uncommon case (fan-out) requires explicit action.

### Why Clear `parallel_tasks` in Merge Instead of in Supervisor?

The supervisor writes `parallel_tasks` as part of its routing decision. If the merge didn't clear it, the next time the supervisor chooses sequential research, `_research_exit` would still see the stale `parallel_tasks` and route to merge incorrectly.

Clearing in merge (the consumer) rather than in the supervisor (the producer) ensures the flag is always cleaned up after use, regardless of what the supervisor does next.

### Why `InMemorySaver` Instead of a Database?

Simplicity. `InMemorySaver` requires zero configuration — no database server, no schema migrations, no connection strings. State persists within a single server process, which is sufficient for the MCP use case (the server runs as a subprocess of the IDE).

The tradeoff: state is lost when the process restarts. For production, swap in `SqliteSaver` (file-based, no server) or `PostgresSaver` (shared state across processes).

### Why Haiku for Both Supervisor and Validator?

Both tasks are "meta" — they reason about outputs, not produce domain knowledge. The supervisor reads summaries and picks a node. The validator reads output and assigns a score. Neither needs deep domain expertise or large context windows.

Haiku is the optimal choice: fast (low latency per supervisor loop), cheap (runs many times per chain), and competent at structured output and evaluation tasks.
