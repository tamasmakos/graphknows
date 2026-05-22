"""Long-term memory recall: Neo4j entity graph + ltm_chunks ANN + RRF."""

from __future__ import annotations

import logging
from typing import Any

from graphknows.memory.ingestor import _run
from graphknows.retrieval.recall_stm import RRF_K, RecallResult

log = logging.getLogger("graphknows.retrieval.recall_ltm")


def recall_long_term(
    neo_store: Any,
    pg_store: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
    embed_fn: Any | None = None,
) -> list[RecallResult]:
    """Recall from long-term memory: Neo4j + ltm_chunks ANN + RRF.

    Args:
        neo_store: Neo4jStore or raw BoltDriver for entity graph traversal.
        pg_store: PgVectorStore with ltm_chunks.
        query: Natural-language query.
        top_k: Max results to return.
        session_filter: Optional session_id to scope LTM results.
        embed_fn: Embedding callable (defaults to embed_one).

    Returns:
        List of RecallResult (scope="long_term") sorted by RRF score descending.
    """
    if embed_fn is None:
        from graphknows.retrieval.stores.embedder import embed_one

        embed_fn = embed_one

    embedding = embed_fn(query)
    words = [w for w in query.lower().split() if len(w) > 3]
    results: dict[str, tuple[float, RecallResult]] = {}

    # ANN search ltm_chunks
    try:
        chunks = _run(
            pg_store.search_ltm_chunks(embedding, top_k=top_k * 2, session_id=session_filter)
        )
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
                    heading_path=chunk.get("heading_path", ""),
                    provenance={
                        "session_id": chunk.get("session_id", ""),
                        "file_id": chunk.get("file_id", ""),
                    },
                ),
            )
    except Exception as exc:
        log.warning("ltm_chunks ANN failed: %s", exc)

    # Entity-first traversal in Neo4j — uses the driver's _driver attribute if wrapped
    if words:
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
                params: dict[str, Any] = {"words": words, "top_k": top_k * 2}
                if session_filter:
                    base_cypher = base_cypher.replace(
                        "MATCH (e:ENTITY)",
                        "MATCH (e:ENTITY {session_id: $sid})",
                    )
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
                            ),
                        )
        except Exception as exc:
            log.warning("Neo4j entity traversal failed: %s", exc)

    sorted_results = sorted(results.values(), key=lambda x: x[0], reverse=True)
    return [r for _, r in sorted_results[:top_k]]
