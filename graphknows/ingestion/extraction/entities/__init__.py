"""Entity extraction — GLiNER NER, canonicalization, and ontology label spaces.

Shared by both memory modes: entity extraction is always GLiNER2-driven.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_entity_extractor(settings: GraphKnowsSettings) -> Any:
    """Return the GLiNER2 entity extractor (same for both modes)."""
    from graphknows.ingestion.extraction.entities.extractor import GLiNER2EntityExtractor

    return GLiNER2EntityExtractor()


__all__ = ["build_entity_extractor"]
