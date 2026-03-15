"""Orchestrator state — shared TypedDict flowing through the LangGraph graph.

Each node reads from and writes to this state. The `messages` field uses
LangGraph's `add_messages` reducer so chat history accumulates across nodes.
Domain output fields are last-writer-wins (str) for easy consumption.
The `output_versions` list uses an append reducer to accumulate every version
across attempts and timelines — useful for comparing rewound branches.
"""

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class OrchestratorState(TypedDict, total=False):
    """State that flows through the orchestrator graph.

    Required fields (set at graph entry):
        task: The user's task description.

    Optional fields (set by nodes as they execute):
        context: User-provided context.
        messages: Chat message history — uses add_messages reducer.

    Domain outputs (set by domain nodes):
        research_findings: Markdown output from the research node.
        architecture_plan: Markdown output from the architect node.
        implementation_result: Output from the implement node.

    Supervisor fields (v0.4 — dynamic supervisor pattern):
        next_node: Which node the supervisor wants to call next.
        supervisor_rationale: Why the supervisor chose that node.
        supervisor_instructions: Instructions for the next node.
        history: List of step summaries (supervisor decisions, node results).
        node_calls: Dict tracking how many times each node has been called.

    Fan-out fields (v0.5 — parallel research):
        parallel_tasks: List of sub-tasks for fan-out. Each dict has 'topic'
            and 'instructions'. Set by supervisor, cleared by merge_research.
        parallel_task_topic: Topic label for a single Send() branch. Set in
            the Send payload, read by the research node.

    Human-in-the-loop fields (v0.5 — HITL):
        human_review_status: "pending", "approved", or "rejected".
            Set by human_review node, updated on resume.
        human_feedback: Feedback from the human reviewer. Empty if approved
            without comment. Passed to the next node on resume.

    Output versioning:
        output_versions: Append-only list of every domain output across
            attempts/timelines. Each entry is a dict with node, attempt,
            and content. Uses operator.add reducer (like messages).

    Validation fields:
        validation_score: 0.0-1.0 quality score from the validator.
        validation_feedback: Actionable feedback if score is low.

    """

    # --- Input ---
    task: str
    context: str

    # --- Chat history ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Domain outputs (latest — last-writer-wins) ---
    research_findings: str
    architecture_plan: str
    implementation_result: str

    # --- Output versioning (append reducer — accumulates across attempts/timelines) ---
    output_versions: Annotated[list[dict[str, Any]], operator.add]

    # --- Supervisor (v0.4) ---
    next_node: str
    supervisor_rationale: str
    supervisor_instructions: str
    history: list[str]
    node_calls: dict[str, int]

    # --- Fan-out (v0.5) ---
    parallel_tasks: list[dict[str, str]]
    parallel_task_topic: str

    # --- Human-in-the-loop (v0.5) ---
    human_review_status: str
    human_feedback: str

    # --- Validation ---
    validation_score: float
    validation_feedback: str

