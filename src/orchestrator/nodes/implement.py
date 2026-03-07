"""Implement node — placeholder for v0.3 (Codex/GPT-4o integration).

Structurally present in the graph but routes to END immediately.
Returns a static message indicating implementation is not yet available.
"""

from ..state import OrchestratorState


def build_implement_node():
    """Build a placeholder implement node.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """

    async def implement_node(state: OrchestratorState) -> dict:
        """Placeholder — returns a static 'not yet available' message."""
        return {
            "implementation_result": (
                "Implementation node is not yet available (v0.3). "
                "Use the architecture plan above to implement manually."
            )
        }

    return implement_node
