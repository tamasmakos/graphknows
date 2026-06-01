"""Interfaces for ontology providers used by ingestion and graph generation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OntologyProvider(Protocol):
    """Protocol for swappable document-type ontology providers."""

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    def supports(self, doc_type: str | None = None, text: str = "") -> bool:
        """Return True when this provider can serve the document."""
        ...

    def class_labels(self) -> list[str]:
        """Return ontology class labels available to entity extraction."""
        ...

    def property_labels(self) -> list[str]:
        """Return ontology property labels available to relation extraction."""
        ...

    def relation_types(self) -> list[str]:
        """Return relation labels available to graph generation."""
        ...

    def summary_jsonld(self) -> str:
        """Return compact ontology context for LLM prompts."""
        ...
