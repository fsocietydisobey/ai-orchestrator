"""LangGraph StateGraph construction — wires classify → research → architect.

The graph routes tasks through a pipeline determined by the classify node:
    - research tasks: classify → research → architect → END
    - architect tasks: classify → architect → END
    - implement tasks: classify → END (placeholder, returns static message)

No checkpointer — each invocation is stateless.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .config import OrchestratorConfig
from .models import get_architect_model, get_classify_model, get_research_model
from .nodes import (
    build_architect_node,
    build_classify_node,
    build_implement_node,
    build_research_node,
)
from .state import OrchestratorState
from .tools import READ_TOOLS


def route_after_classify(
    state: OrchestratorState,
) -> Literal["research", "architect", "end"]:
    """Route to the next node based on the classification pipeline.

    Reads the `pipeline` field from classification and returns the first
    relevant step: 'research' if research is needed, 'architect' if only
    architecture is needed, or 'end' for pure implement tasks.
    """
    classification = state.get("classification", {})
    pipeline = classification.get("pipeline", ["architect", "implement"])

    if "research" in pipeline:
        return "research"
    elif "architect" in pipeline:
        return "architect"
    else:
        # Pure implement or unknown — go to end
        return "end"


def build_orchestrator_graph(config: OrchestratorConfig):
    """Build and compile the orchestrator StateGraph.

    Creates LangChain models from config, builds node functions with tools,
    and wires the graph edges.

    Args:
        config: OrchestratorConfig with provider/role definitions.

    Returns:
        Compiled StateGraph ready for .ainvoke().
    """
    # --- Build models from config ---
    classify_model = get_classify_model(config)
    research_model = get_research_model(config)
    architect_model = get_architect_model(config)

    # --- Build node functions ---
    classify_node = build_classify_node(classify_model)
    research_node = build_research_node(research_model, READ_TOOLS)
    architect_node = build_architect_node(architect_model, READ_TOOLS)
    implement_node = build_implement_node()

    # --- Wire the graph ---
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("research", research_node)
    graph.add_node("architect", architect_node)
    graph.add_node("implement", implement_node)

    # Entry edge: START → classify
    graph.add_edge(START, "classify")

    # Conditional routing after classify
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "research": "research",
            "architect": "architect",
            "end": "implement",  # placeholder node, then END
        },
    )

    # Sequential edges: research → architect, architect → END, implement → END
    graph.add_edge("research", "architect")
    graph.add_edge("architect", END)
    graph.add_edge("implement", END)

    # Compile without checkpointer (stateless MCP calls)
    return graph.compile()
