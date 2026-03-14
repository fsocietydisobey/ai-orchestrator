"""Orchestrator state — shared TypedDict flowing through the LangGraph graph.

Each node reads from and writes to this state. The `messages` field uses
LangGraph's `add_messages` reducer so chat history accumulates across nodes.
Other fields are simple overwrites (last writer wins).
"""

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

    Validation fields:
        validation_score: 0.0-1.0 quality score from the validator.
        validation_feedback: Actionable feedback if score is low.

    Legacy fields (v0.3 — kept for backward compatibility):
        classification: Dict from the classify node (tier, confidence, etc.).
        research_score: Quality score from research critique.
        research_critique: Feedback from research critique.
        research_attempts: Number of research attempts.
        architect_score: Quality score from architect critique.
        architect_critique: Feedback from architect critique.
        architect_attempts: Number of architect attempts.
    """

    # --- Input ---
    task: str
    context: str

    # --- Chat history ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Domain outputs ---
    research_findings: str
    architecture_plan: str
    implementation_result: str

    # --- Supervisor (v0.4) ---
    next_node: str
    supervisor_rationale: str
    supervisor_instructions: str
    history: list[str]
    node_calls: dict[str, int]

    # --- Validation ---
    validation_score: float
    validation_feedback: str

    # --- Legacy (v0.3) ---
    classification: dict[str, Any]
    research_score: float
    research_critique: str
    research_attempts: int
    architect_score: float
    architect_critique: str
    architect_attempts: int
