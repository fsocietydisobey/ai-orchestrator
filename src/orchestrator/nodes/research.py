"""Research node — ReAct agent with filesystem tools for deep exploration.

Uses create_react_agent to give the research model the ability to read files,
search code, and explore directories. The agent autonomously decides which
files to read based on the task description.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from ..prompts import RESEARCH_SYSTEM_PROMPT
from ..state import OrchestratorState


def build_research_node(model: BaseChatModel, tools: list[BaseTool]):
    """Build a research node backed by a ReAct agent with filesystem tools.

    The agent receives the task description, uses tools to explore the
    codebase, and returns structured research findings.

    Args:
        model: LangChain chat model configured for research.
        tools: List of filesystem tools the agent can use.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """
    # Build a standalone ReAct agent (not compiled with a checkpointer)
    research_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=RESEARCH_SYSTEM_PROMPT,
    )

    async def research_node(state: OrchestratorState) -> dict:
        """Run the research agent and return findings."""
        task = state.get("task", "")
        context = state.get("context", "")

        # Build the user message
        prompt = task
        if context:
            prompt = f"{task}\n\n## Context\n\n{context}"

        # Invoke the ReAct agent
        result = await research_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # Extract the final AI message content as research findings
        findings = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            findings = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else str(last_msg.content)
            )

        return {"research_findings": findings}

    return research_node
