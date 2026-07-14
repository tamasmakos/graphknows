"""Ingestion — the write path: parse -> preprocess -> extract -> STM -> promote.

Exposes the input parsers and the settings-driven factories that assemble the
STM service, the promotion pipeline, and the corpus (batch) workflow. The mode
fork lives inside the injected collaborators (relation extractor, promotion
steps), never in this package's callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.ingestion.parsers.base import BaseParser, ParsedFile, TextSegment
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.ingestion.promotion import build_promotion_pipeline

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_stm_service(settings: GraphKnowsSettings, namespace: str = "") -> Any:
    """Build an STMService over the golden GraphStore (unconnected).

    Returned without an open connection so the ArcadeDB client binds to the
    caller's event loop when ``connect()`` is awaited, not to a background loop.
    """
    from graphknows.ingestion.extraction.entities import build_entity_extractor
    from graphknows.ingestion.extraction.relations import build_relation_extractor
    from graphknows.ingestion.stm.service import STMService
    from graphknows.storage import build_graph_store
    from graphknows.storage.embedder import embed_one

    return STMService(
        build_graph_store(settings, namespace),
        build_entity_extractor(settings),
        embed_fn=embed_one,
        rel_extractor=build_relation_extractor(settings),
        embed_dims=settings.embed_dimensions,
        enable_frames=settings.enable_frames,
    )


def build_corpus_workflow(settings: GraphKnowsSettings, namespace: str = "") -> Any:
    """Build a CorpusIngestWorkflow (batch document ingestion)."""
    from graphknows.channels.registry import default_channels
    from graphknows.ingestion.extraction.entities import build_entity_extractor
    from graphknows.ingestion.workflows.corpus_ingest import CorpusIngestWorkflow
    from graphknows.settings import MemoryMode
    from graphknows.storage import (
        build_arcadedb_ltm_store,
        build_arcadedb_stm_store,
        build_arcadedb_vector_store,
    )
    from graphknows.storage.embedder import embed_one

    return CorpusIngestWorkflow(
        stm_store=build_arcadedb_stm_store(settings, namespace),
        vec_store=build_arcadedb_vector_store(settings, namespace),
        ltm_store=build_arcadedb_ltm_store(settings, namespace),
        extractor=build_entity_extractor(settings),
        embed_fn=embed_one,
        llm_api_key=settings.llm_api_key or "",
        llm_api_base=settings.llm_api_base or "",
        rel_extractor=None,
        extraction_uses_llm=settings.mode is MemoryMode.llm_assisted,
        flush_vec_store=build_arcadedb_vector_store(settings, namespace),
        channels=default_channels(settings),
    )


__all__ = [
    "BaseParser",
    "ParsedFile",
    "TextSegment",
    "build_corpus_workflow",
    "build_promotion_pipeline",
    "build_stm_service",
    "get_parser",
]
