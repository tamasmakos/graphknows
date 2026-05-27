"""ILlm — Protocol for LLM completion operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ILlm(Protocol):
    """Protocol for language model completion (minimal interface)."""

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: Input prompt string.
            **kwargs: Additional model-specific parameters (temperature,
                max_tokens, etc.).

        Returns:
            Completion text as a string.
        """
        ...
