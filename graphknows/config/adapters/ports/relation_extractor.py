"""IRelationExtractor — Protocol for async relation extraction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRelationExtractor(Protocol):
    """Protocol for text-to-graph relation extraction."""

    async def extract_relations(
        self,
        text: str,
        keywords: list[str] | None = None,
        entities: list[str] | None = None,
        ontology_classes: list[str] | None = None,
        allowed_relation_types: list[str] | None = None,
        candidate_relations: list[dict[str, Any]] | None = None,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract (source, relation, target, props) triplets from text."""
        ...
