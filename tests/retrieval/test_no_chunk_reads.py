"""After cutover the retriever reads segments and facts, never the old chunk plane (FR-045)."""

from __future__ import annotations

import inspect

import graphknows.retrieval.retriever as retriever

_OLD_READS = (
    "search_chunks_ann",
    "full_text_search_chunks",
    "query_temporal_chunks",
    "query_dated_chunks_by_entities",
    "neighbor_chunks",
    "get_chunks_by_ids",
    "query_entity_chunks",
)


def test_retriever_has_no_chunk_reads() -> None:
    source = inspect.getsource(retriever)
    leaked = [name for name in _OLD_READS if name in source]
    assert not leaked, f"retriever still reads the dropped chunk plane: {leaked}"
