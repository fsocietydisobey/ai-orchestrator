"""Critique nodes — cheap model scores output quality, loops back if weak.

Uses the classify model (Haiku) to evaluate research findings and architecture
plans. If the score is below threshold, the critique is fed back to the
originating node for a retry.
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import OrchestratorState

CRITIQUE_SYSTEM_PROMPT = """\
You are a quality reviewer. Score the following output on a 0.0-1.0 scale
and provide brief, actionable feedback if the score is below 0.7.

Respond with ONLY a JSON object — no other text.

## Scoring criteria

- **Completeness**: Does it address the full scope of the task?
- **Specificity**: Does it reference concrete details (file paths, function names, patterns)?
- **Actionability**: Could someone act on this without asking follow-up questions?
- **Accuracy**: Does it avoid hallucinations or vague hand-waving?

## Response format

{
  "score": 0.0-1.0,
  "feedback": "Brief, actionable feedback if score < 0.7. Empty string if score >= 0.7."
}
"""

# Minimum acceptable quality score
QUALITY_THRESHOLD = 0.7
# Max retry attempts per node
MAX_ATTEMPTS = 2


def build_research_critique_node(model: BaseChatModel):
    """Build a critique node that evaluates research findings."""

    async def research_critique_node(state: OrchestratorState) -> dict:
        findings = state.get("research_findings", "")
        task = state.get("task", "")
        attempts = state.get("research_attempts", 1)

        prompt = (
            f"## Task\n\n{task}\n\n"
            f"## Output to evaluate (research findings)\n\n{findings}"
        )

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await model.ainvoke(messages)
        raw = response.content.strip()

        # Parse JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
            feedback = result.get("feedback", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            feedback = "Failed to parse critique response."

        return {
            "research_score": score,
            "research_critique": feedback,
            "research_attempts": attempts,
        }

    return research_critique_node


def build_architect_critique_node(model: BaseChatModel):
    """Build a critique node that evaluates architecture plans."""

    async def architect_critique_node(state: OrchestratorState) -> dict:
        plan = state.get("architecture_plan", "")
        task = state.get("task", "")
        research = state.get("research_findings", "")
        attempts = state.get("architect_attempts", 1)

        prompt = (
            f"## Task\n\n{task}\n\n"
            f"## Research context\n\n{research[:2000]}\n\n"
            f"## Output to evaluate (architecture plan)\n\n{plan}"
        )

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await model.ainvoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
            feedback = result.get("feedback", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            feedback = "Failed to parse critique response."

        return {
            "architect_score": score,
            "architect_critique": feedback,
            "architect_attempts": attempts,
        }

    return architect_critique_node
