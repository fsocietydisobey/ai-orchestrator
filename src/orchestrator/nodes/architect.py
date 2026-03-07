"""Architect node — ReAct agent with filesystem tools for design/planning.

Uses create_react_agent to give the architect model file-reading capability.
If research findings exist in state, they're included in the prompt so the
architect builds on prior exploration.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from ..prompts import ARCHITECT_SYSTEM_PROMPT
from ..state import OrchestratorState


def build_architect_node(model: BaseChatModel, tools: list[BaseTool]):
    """Build an architect node backed by a ReAct agent with filesystem tools.

    The agent receives the task description (plus research findings if
    available), reads relevant code files, and produces a structured
    implementation plan.

    Args:
        model: LangChain chat model configured for architecture.
        tools: List of filesystem tools the agent can use.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """
    architect_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=ARCHITECT_SYSTEM_PROMPT,
    )

    async def architect_node(state: OrchestratorState) -> dict:
        """Run the architect agent and return the implementation plan."""
        task = state.get("task", "")
        context = state.get("context", "")
        research_findings = state.get("research_findings", "")

        # Build the user message — include context and research findings
        parts = [task]
        if context:
            parts.append(f"## Context\n\n{context}")
        if research_findings:
            parts.append(f"## Research Findings\n\n{research_findings}")
        prompt = "\n\n".join(parts)

        # Invoke the ReAct agent
        result = await architect_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # Extract the final AI message content as the architecture plan
        plan = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            plan = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else str(last_msg.content)
            )

        return {"architecture_plan": plan}

    return architect_node
