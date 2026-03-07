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
        context: User-provided context (file contents, prior findings).
        classification: Dict from the classify node (tier, confidence, etc.).
        messages: Chat message history — uses add_messages reducer.
        research_findings: Markdown output from the research node.
        architecture_plan: Markdown output from the architect node.
        implementation_result: Output from the implement node (placeholder).
    """

    # --- Input (set before graph.ainvoke) ---
    task: str
    context: str

    # --- Set by classify node ---
    classification: dict[str, Any]

    # --- Chat history (accumulates via add_messages reducer) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Set by domain nodes ---
    research_findings: str
    architecture_plan: str
    implementation_result: str
