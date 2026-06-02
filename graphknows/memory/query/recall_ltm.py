"""Long-term memory recall: Neo4j entity graph + ltm_chunks ANN + ltm_turns ANN + RRF."""

from __future__ import annotations

import logging
from typing import Any

from graphknows.memory.models.recall import RecallResult
from graphknows.memory.query.scoring import RRF_K, _shape_score
from graphknows.memory.stm._async_bridge import _run

log = logging.getLogger("graphknows.memory.query.recall_ltm")


def _ann_chunks(
    pg_store: Any,
    embedding: list[float],
    top_k: int,
    session_filter: str | None,
) -> dict[str, tuple[float, RecallResult]]:
    """ANN search on ltm_chunks; returns {id: (rrf_score, RecallResult)}."""
    results: dict[str, tuple[float, RecallResult]] = {}
    try:
        chunks = _run(pg_store.search_ltm_chunks(embedding, top_k=top_k, session_id=session_filter))
        for rank, chunk in enumerate(chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = chunk["id"]
            results[cid] = (
                rrf,
                RecallResult(
                    text=chunk["text"],
                    source_type="chunk",
                    scope="long_term",
                    score=rrf,
                    cosine_score=chunk.get("cosine_score", 0.0),
                    heading_path=chunk.get("heading_path", ""),
                    provenance={
                        "session_id": chunk.get("session_id", ""),
                        "file_id": chunk.get("file_id", ""),
                    },
                    metadata=chunk.get("metadata", {}),
                ),
            )
    except Exception as exc:
        log.warning("ltm_chunks ANN failed: %s", exc)
    return results


def _ann_turns(
    pg_store: Any,
    embedding: list[float],
    top_k: int,
    session_filter: str | None,
) -> list[RecallResult]:
    """ANN search on ltm_turns; returns RecallResult list for interaction/dialog recall."""
    turn_results: list[RecallResult] = []
    try:
        turns = _run(pg_store.search_ltm_turns(embedding, top_k=top_k, session_id=session_filter))
        for rank, turn in enumerate(turns):
            rrf = 1.0 / (RRF_K + rank + 1)
            turn_results.append(
                RecallResult(
                    text=turn["text"],
                    source_type="turn",
                    scope="long_term",
                    score=rrf,
                    cosine_score=turn.get("cosine_score", 0.0),
                    provenance={
                        "session_id": turn.get("session_id", ""),
                        "dialog_id": turn.get("dialog_id", ""),
                        "turn_index": turn.get("turn_index"),
                        "role": turn.get("role", ""),
                    },
                    metadata=turn.get("metadata", {}),
                )
            )
    except Exception as exc:
        log.warning("ltm_turns ANN failed: %s", exc)
    return turn_results


def _entity_traversal(
    neo_store: Any,
    words: list[str],
    top_k: int,
    session_filter: str | None,
    existing: dict[str, tuple[float, RecallResult]],
) -> dict[str, tuple[float, RecallResult]]:
    """Entity-first traversal in Neo4j; RRF-merges into existing chunk results dict."""
    if not words:
        return existing
    results = dict(existing)
    try:
        driver = getattr(neo_store, "_driver", neo_store)
        with driver.session() as s:
            base_cypher = """
                MATCH (e:ENTITY)
                WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                MATCH (c:CHUNK)-[:HAS_ENTITY]->(e)
                WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                ORDER BY mc DESC LIMIT $top_k
                RETURN c.id AS chunk_id, c.text AS text, entities
            """
            params: dict[str, Any] = {"words": words, "top_k": top_k}
            if session_filter:
                base_cypher = """
                    MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->()-[*0..]->(c:CHUNK)
                    MATCH (c)-[:HAS_ENTITY]->(e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                """
                params["sid"] = session_filter
            rows = s.run(base_cypher, **params).data()
            for rank, row in enumerate(rows):
                rrf = 1.0 / (RRF_K + rank + 1)
                cid = row["chunk_id"]
                if cid in results:
                    old_score, old_result = results[cid]
                    new_score = old_score + rrf
                    old_result.score = new_score
                    old_result.entity_names = row["entities"]
                    results[cid] = (new_score, old_result)
                else:
                    results[cid] = (
                        rrf,
                        RecallResult(
                            text=row["text"],
                            source_type="chunk",
                            scope="long_term",
                            score=rrf,
                            entity_names=row["entities"],
                            provenance={},
                            metadata=row.get("metadata", {}),
                        ),
                    )
    except Exception as exc:
        log.warning("Neo4j entity traversal failed: %s", exc)
    return results


def recall_long_term(
    neo_store: Any,
    pg_store: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
    embed_fn: Any | None = None,
) -> list[RecallResult]:
    """Recall from long-term memory: Neo4j entity graph + ltm_chunks + ltm_turns ANN + RRF.

    Args:
        neo_store: Neo4jStore or raw BoltDriver for entity graph traversal.
        pg_store: PgVectorStore with ltm_chunks and ltm_turns tables.
        query: Natural-language query.
        top_k: Max results to return.
        session_filter: Optional session_id to scope LTM results.
        embed_fn: Embedding callable (defaults to embed_one).

    Returns:
        List of RecallResult (scope="long_term") sorted by RRF score descending.
        Includes both document chunks (source_type="chunk") and dialog turns
        (source_type="turn") — enabling ActivityLocal/Global interaction recall.
    """
    if embed_fn is None:
        from graphknows.memory.stores.embedder import embed_one

        embed_fn = embed_one

    embedding = embed_fn(query)
    words = [w for w in query.lower().split() if len(w) > 3]

    chunk_results = _ann_chunks(pg_store, embedding, top_k * 2, session_filter)
    chunk_results = _entity_traversal(neo_store, words, top_k * 2, session_filter, chunk_results)
    turn_results = _ann_turns(pg_store, embedding, top_k * 2, session_filter)

    # RRF-merge chunks and turns — turns carry provenance (role, turn_index, dialog_id)
    combined: dict[int, tuple[float, RecallResult]] = {}
    for score, r in chunk_results.values():
        combined[id(r)] = (score, r)
    for r in turn_results:
        combined[id(r)] = (r.score, r)

    final = sorted(combined.values(), key=lambda x: x[0], reverse=True)
    shaped = []
    for score, result in final:
        memory_class = result.metadata.get("memory_class")
        new_score = _shape_score(score, memory_class)
        result.score = new_score
        shaped.append((new_score, result))
    shaped.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in shaped[:top_k]]
