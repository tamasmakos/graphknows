"""IEmbedder — Protocol for text embedding operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IEmbedder(Protocol):
    """Protocol for text-to-vector embedding operations."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one float vector per text.

        Args:
            texts: Input strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return a plain float list.

        Args:
            text: Input string.

        Returns:
            Embedding as a plain Python list of floats.
        """
        ...
