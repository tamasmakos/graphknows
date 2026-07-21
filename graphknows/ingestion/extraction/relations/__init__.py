"""Relation extraction — capability-forked behind one factory.

``build_relation_extractor(settings)`` is the single place the LLM relation
strategy is selected: with ``settings.dspy_relations`` it returns the
DSPy/ontology adapter (lazily imported so ``dspy``/``rdflib`` load only then),
otherwise :class:`NullRelationExtractor`.

This is *additive* — the deterministic relex/SVO relation pass runs in both
modes regardless, so what this factory gates is the LLM layer on top.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from graphknows.ingestion.extraction.relations.base import (
    IRelationExtractor,
    NullRelationExtractor,
)

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_relation_extractor(settings: GraphKnowsSettings) -> IRelationExtractor:
    """Return the LLM relation extractor when enabled, else the null object."""
    if not settings.dspy_relations:
        return NullRelationExtractor()

    from graphknows.ingestion.extraction.relations.llm_assisted.adapter import (
        DspyRelationExtractorAdapter,
    )

    ontology_file = settings.ontology_source or os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE", "")
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
