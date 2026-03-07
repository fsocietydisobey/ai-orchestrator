"""Classify node — fast/cheap LLM call to determine task tier and pipeline.

No tools needed. The classifier receives the task description and returns
a JSON dict with tier, confidence, reasoning, and pipeline fields.
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..state import OrchestratorState


def build_classify_node(model: BaseChatModel):
    """Build a classify node function for the orchestrator graph.

    The node sends the task to a cheap/fast model with the classifier prompt,
    parses the JSON response, and writes classification to state.

    Args:
        model: LangChain chat model configured for classification.

    Returns:
        Async node function compatible with LangGraph StateGraph.
    """

    async def classify_node(state: OrchestratorState) -> dict:
        """Classify the task and return the classification dict."""
        task = state.get("task", "")
        context = state.get("context", "")

        # Build the prompt — include context if provided
        prompt = task
        if context:
            prompt = f"{task}\n\n## Context\n\n{context}"

        # Call the model
        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await model.ainvoke(messages)
        raw = response.content

        # Parse JSON from the response (strip markdown fences if present)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            classification = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: default to architect pipeline
            classification = {
                "tier": "architect",
                "confidence": 0.5,
                "reasoning": "Failed to parse classifier response, defaulting to architect.",
                "pipeline": ["architect", "implement"],
                "raw_response": raw,
            }

        return {"classification": classification}

    return classify_node
