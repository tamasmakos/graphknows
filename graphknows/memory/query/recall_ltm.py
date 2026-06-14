"""Long-term memory recall: Neo4j entity graph + ltm_chunks ANN + ltm_turns ANN + RRF."""

from __future__ import annotations

import logging
from typing import Any

from graphknows.memory.models.recall import RecallResult
from graphknows.memory.query._rrf_common import RRF_K
from graphknows.memory.query._rrf_common import shape_score as _shape_score
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


_STRUCTURAL_REL_TYPES = frozenset(
    {
        "MENTIONS",
        "HAS_CHUNK",
        "IN_SESSION",
        "IN_TOPIC",
        "IN_COMMUNITY",
        "HAS_TURN",
        "IN_DIALOG",
        "PART_OF",
    }
)


def _entity_traversal(
    neo_store: Any,
    words: list[str],
    top_k: int,
    session_filter: str | None,
    existing: dict[str, tuple[float, RecallResult]],
) -> dict[str, tuple[float, RecallResult]]:
    """Entity-first traversal in Neo4j; RRF-merges into existing chunk results dict.

    Two passes:
    1. Direct: query words → matched ENTITY → CHUNK (weight 1.0).
    2. Relation-expanded: matched ENTITY → semantic RELATION edge → neighbor ENTITY
       → CHUNK (weight 0.5). Uses edges written by llm_assisted extraction.
    """
    if not words:
        return existing
    results = dict(existing)
    try:
        driver = getattr(neo_store, "_driver", neo_store)
        _db = getattr(neo_store, "_db", None)
        _session_kwargs: dict[str, Any] = {"database": _db} if _db else {}
        with driver.session(**_session_kwargs) as s:
            # Pass 1 — direct entity match → chunk
            if session_filter:
                direct_cypher = """
                    MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
                    MATCH (c)-[:MENTIONS]->(e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                """
                params: dict[str, Any] = {"words": words, "top_k": top_k, "sid": session_filter}
            else:
                direct_cypher = """
                    MATCH (e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    MATCH (c:CHUNK)-[:MENTIONS]->(e)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                """
                params = {"words": words, "top_k": top_k}

            for rank, row in enumerate(s.run(direct_cypher, **params).data()):
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

            # Pass 2 — relation-expanded: matched entity → semantic relation → neighbor → chunk
            # Only fires when llm_assisted relation edges exist (no-op for gliner2).
            try:
                if session_filter:
                    rel_cypher = """
                        MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c0:CHUNK)
                        MATCH (c0)-[:MENTIONS]->(e:ENTITY)
                        WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                        MATCH (e)-[r]-(e2:ENTITY)
                        WHERE NOT type(r) IN $skip_types
                        MATCH (sess)-[:IN_SESSION]->(f2:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS]->(e2)
                        WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc
                        ORDER BY mc DESC LIMIT $top_k
                        RETURN c.id AS chunk_id, c.text AS text, ents AS entities
                    """
                    rel_params: dict[str, Any] = {
                        "words": words,
                        "top_k": top_k,
                        "sid": session_filter,
                        "skip_types": list(_STRUCTURAL_REL_TYPES),
                    }
                else:
                    rel_cypher = """
                        MATCH (e:ENTITY)
                        WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                        MATCH (e)-[r]-(e2:ENTITY)
                        WHERE NOT type(r) IN $skip_types
                        MATCH (c:CHUNK)-[:MENTIONS]->(e2)
                        WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc
                        ORDER BY mc DESC LIMIT $top_k
                        RETURN c.id AS chunk_id, c.text AS text, ents AS entities
                    """
                    rel_params = {
                        "words": words,
                        "top_k": top_k,
                        "skip_types": list(_STRUCTURAL_REL_TYPES),
                    }
                for rank, row in enumerate(s.run(rel_cypher, **rel_params).data()):
                    rrf = 1.5 / (RRF_K + rank + 1)  # boosted weight for relation-traversal results
                    cid = row["chunk_id"]
                    if cid in results:
                        old_score, old_result = results[cid]
                        results[cid] = (old_score + rrf, old_result)
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
            except Exception as rel_exc:
                log.debug("relation-expanded traversal skipped: %s", rel_exc)

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
