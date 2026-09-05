"""Ingestion — the write path: parse -> preprocess -> extract -> STM.

Exposes the parser's output types and the settings-driven factory that assembles
the STM service. The mode fork lives inside the injected decoder, never in this
package's callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.ingestion.parsers.text import ParsedFile, PlainTextParser, TextSegment

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_stm_service(settings: GraphKnowsSettings, namespace: str = "") -> Any:
    """Build an STMService over the golden GraphStore (unconnected).

    Returned without an open connection so the ArcadeDB client binds to the
    caller's event loop when ``connect()`` is awaited, not to a background loop.
    """
    from graphknows.ingestion.extraction.entities import build_decoder
    from graphknows.ingestion.stm.service import STMService
    from graphknows.storage import build_graph_store
    from graphknows.storage.embedder import embed_one
    from graphknows.symbolic.ontology import load_ontology_index

    return STMService(
        build_graph_store(settings, namespace),
        build_decoder(settings),
        embed_fn=embed_one,
        embed_dims=settings.embed_dimensions,
        enable_frames=settings.enable_frames,
        # Mode-independent and always on: ontology injection steers extraction
        # identically in llm_free and llm_assisted, and ``ontology_source``
        # falls back to the bundled CCO digest. ``None`` here therefore
        # means the load FAILED, which load_ontology_index logs.
        ontology_index=load_ontology_index(settings.ontology_source),
    )


__all__ = [
    "ParsedFile",
    "PlainTextParser",
    "TextSegment",
    "build_stm_service",
]
