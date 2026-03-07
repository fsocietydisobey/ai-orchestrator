"""Anthropic (Claude) provider."""

import os

from anthropic import AsyncAnthropic

from .base import Provider


class AnthropicProvider(Provider):
    """Routes requests to Claude via the Anthropic API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @property
    def name(self) -> str:
        return f"Claude ({self._model})"

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Send a message to Claude and return the text response."""
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt if system_prompt else [],
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from content blocks
        return "".join(
            block.text for block in message.content if hasattr(block, "text")
        )
