"""IEntityExtractor — Protocol for named entity extraction operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IEntityExtractor(Protocol):
    """Protocol for named entity extraction (GLiNER2 boundary)."""

    def extract(self, text: str) -> list[dict[str, object]]:
        """Extract named entities from text.

        Args:
            text: Input text to extract entities from.

        Returns:
            List of entity dicts (keys vary by implementation, typically
            include 'text', 'label', 'score').
        """
        ...
