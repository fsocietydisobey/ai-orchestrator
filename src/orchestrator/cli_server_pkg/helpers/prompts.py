"""Prompt building helpers."""


def build_prompt(*parts: str) -> str:
    """Join non-empty prompt parts with double newlines."""
    return "\n\n".join(p for p in parts if p)
