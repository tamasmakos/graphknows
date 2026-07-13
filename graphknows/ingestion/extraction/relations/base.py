"""Relation-extraction port + the llm_free null object.

``IRelationExtractor`` is the contract the STM/LTM write path depends on. Both
modes inject an implementation, so ingest code never branches on the mode:
llm_assisted injects the DSPy adapter; llm_free injects :class:`NullRelationExtractor`,
which extracts nothing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRelationExtractor(Protocol):
    """Protocol for text-to-graph relation extraction."""

    async def extract_relations(
        self,
        text: str,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract (source, relation, target, props) triplets from text."""
        ...


class NullRelationExtractor:
    """No-op relation extractor for llm_free mode — extracts nothing.

    Lets the write path call ``extract_relations`` unconditionally instead of
    guarding on ``rel_extractor is not None``.
    """

    async def extract_relations(
        self, text: str
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Return no triplets, no entities, empty metadata."""
        return [], [], {}


__all__ = ["IRelationExtractor", "NullRelationExtractor"]
