"""Relation extraction — capability-forked behind one factory.

``build_relation_extractor(settings)`` is the single place the LLM relation
strategy is selected: under ``Decoder.llm`` it returns the DSPy/ontology
extractor (lazily imported so ``dspy``/``rdflib`` load only then), otherwise
``None``.

This is *additive* — the deterministic relex/SVO relation pass runs in both
modes regardless, so what this factory gates is the LLM layer on top.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from graphknows.settings import Decoder

if TYPE_CHECKING:
    from graphknows.ingestion.extraction.relations.llm_assisted.dspy import DSPyExtractor
    from graphknows.settings import GraphKnowsSettings


def build_relation_extractor(settings: GraphKnowsSettings) -> DSPyExtractor | None:
    """Return the LLM relation extractor when enabled, else ``None``."""
    if settings.decoder is not Decoder.llm:
        return None

    from graphknows.ingestion.extraction.relations.llm_assisted.dspy import DSPyExtractor

    ontology_file = settings.ontology_source or os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE", "")
    return DSPyExtractor(
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


__all__ = ["build_relation_extractor"]
