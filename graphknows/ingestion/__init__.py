"""Ingestion — the write path: parse -> preprocess -> extract -> STM.

Exposes the input parsers and the settings-driven factory that assembles the
STM service. The mode fork lives inside the injected collaborators (relation
extractor), never in this package's callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.ingestion.parsers.base import BaseParser, ParsedFile, TextSegment
from graphknows.ingestion.parsers.registry import get_parser

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_stm_service(settings: GraphKnowsSettings, namespace: str = "") -> Any:
    """Build an STMService over the golden GraphStore (unconnected).

    Returned without an open connection so the ArcadeDB client binds to the
    caller's event loop when ``connect()`` is awaited, not to a background loop.
    """
    from graphknows.ingestion.extraction.entities import build_entity_extractor
    from graphknows.ingestion.extraction.relations import build_relation_extractor
    from graphknows.ingestion.stm.config import STMConfig
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
        config=STMConfig(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap),
    )


__all__ = [
    "BaseParser",
    "ParsedFile",
    "TextSegment",
    "build_stm_service",
    "get_parser",
]
