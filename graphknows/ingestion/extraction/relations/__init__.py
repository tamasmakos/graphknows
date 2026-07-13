"""Relation extraction — mode-forked behind one factory.

``build_relation_extractor(settings)`` is the single place the memory mode
selects a relation strategy: ``llm_assisted`` returns the DSPy/ontology adapter
(lazily imported so ``dspy``/``rdflib`` load only in that mode); ``llm_free``
returns :class:`NullRelationExtractor`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from graphknows.ingestion.extraction.relations.base import (
    IRelationExtractor,
    NullRelationExtractor,
)
from graphknows.settings import MemoryMode

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_relation_extractor(settings: GraphKnowsSettings) -> IRelationExtractor:
    """Return the relation extractor for the active memory mode."""
    if settings.mode is not MemoryMode.llm_assisted:
        return NullRelationExtractor()

    from graphknows.ingestion.extraction.relations.llm_assisted.adapter import (
        DspyRelationExtractorAdapter,
    )

    ontology_file = settings.ontology_file or os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE", "")
    return DspyRelationExtractorAdapter(
        config={
            "llm": {
                "model": settings.extraction_model or settings.llm_model,
                "api_key": settings.llm_api_key or "",
                "api_base": settings.llm_api_base or "",
                "temperature": settings.llm_temperature,
                # Relation extraction needs headroom: reasoning models spend the
                # budget on reasoning_content and truncate the structured triplet
                # output at 2048, producing malformed/empty JSON.
                "max_tokens": max(settings.llm_max_tokens, 4096),
            },
            "ontology": {"file": ontology_file},
        }
    )


__all__ = ["IRelationExtractor", "NullRelationExtractor", "build_relation_extractor"]
