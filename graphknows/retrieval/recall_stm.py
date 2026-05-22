"""Short-term memory recall: Memgraph entity graph + pgvector stm_chunks/stm_turns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from graphknows.memory.ingestor import _run

log = logging.getLogger("graphknows.retrieval.recall_stm")

RRF_K: int = 60


@dataclass
class RecallResult:
    """A single recall result from either STM or LTM."""

    text: str
    source_type: Literal["chunk", "turn"]
    scope: Literal["short_term", "long_term"]
    score: float
    entity_names: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    heading_path: str = ""


def _run_stm_ann(
    pg_store: Any,
    session_id: str,
    embedding: list[float],
    top_k: int,
    results: dict[str, tuple[float, RecallResult]],
) -> None:
    """ANN search stm_chunks + stm_turns; populate results dict."""
    try:
        chunks = _run(pg_store.search_stm_chunks(session_id, embedding, top_k=top_k * 2))
        for rank, chunk in enumerate(chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = chunk["id"]
            if cid not in results or results[cid][0] < rrf:
                results[cid] = (
                    rrf,
                    RecallResult(
                        text=chunk["text"],
                        source_type="chunk",
                        scope="short_term",
                        score=rrf,
                        heading_path=chunk.get("heading_path", ""),
                        provenance={"session_id": session_id, "file_id": chunk.get("file_id", "")},
                    ),
                )
    except Exception as exc:
        log.warning("stm_chunks ANN failed: %s", exc)

    try:
        turns = _run(pg_store.search_stm_turns(session_id, embedding, top_k=top_k * 2))
        for rank, turn in enumerate(turns):
            rrf = 1.0 / (RRF_K + rank + 1)
            tid = turn["id"]
            if tid not in results or results[tid][0] < rrf:
                results[tid] = (
                    rrf,
                    RecallResult(
                        text=turn["text"],
                        source_type="turn",
                        scope="short_term",
                        score=rrf,
                        provenance={"session_id": session_id, "role": turn.get("role", "")},
                    ),
                )
    except Exception as exc:
        log.warning("stm_turns ANN failed: %s", exc)


def _entity_boost(
    mg_store: Any,
    session_id: str,
    words: list[str],
    top_k: int,
    results: dict[str, tuple[float, RecallResult]],
) -> None:
    """Entity-first path: boost chunks co-mentioned with matched entities."""
    if not words:
        return
    try:
        entity_chunks = _run(
            mg_store.query_entity_chunks(
                session_id=session_id,
                entity_names=words,
                top_k=top_k,
            )
        )
        for rank, ec in enumerate(entity_chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = ec["chunk_id"]
            if cid in results:
                old_score, old_result = results[cid]
                new_score = old_score + rrf
                old_result.score = new_score
                old_result.entity_names = ec.get("entity_names", [])
                results[cid] = (new_score, old_result)
    except Exception as exc:
        log.warning("Memgraph entity path failed: %s", exc)


def recall_short_term(
    mg_store: Any,
    pg_store: Any,
    query: str,
    session_id: str,
    top_k: int = 5,
    embed_fn: Any | None = None,
) -> list[RecallResult]:
    """Recall from short-term memory: Memgraph entity graph + stm_* ANN, RRF-fused.

    Args:
        mg_store: MemgraphStore with session entity graph.
        pg_store: PgVectorStore with stm_chunks and stm_turns.
        query: Natural-language query.
        session_id: Session namespace to scope all queries.
        top_k: Max results to return.
        embed_fn: Embedding callable (defaults to embed_one).

    Returns:
        List of RecallResult sorted by RRF score descending.
    """
    if embed_fn is None:
        from graphknows.retrieval.stores.embedder import embed_one

        embed_fn = embed_one

    words = [w for w in query.lower().split() if len(w) > 3]
    embedding = embed_fn(query)
    results: dict[str, tuple[float, RecallResult]] = {}

    _run_stm_ann(pg_store, session_id, embedding, top_k, results)
    _entity_boost(mg_store, session_id, words, top_k, results)

    sorted_results = sorted(results.values(), key=lambda x: x[0], reverse=True)
    return [r for _, r in sorted_results[:top_k]]
