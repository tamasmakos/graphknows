# STM-007: Dual recall — recall_short_term + recall_long_term + HybridRetriever scope routing

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

Depends on: STM-001 (stm_chunks, stm_turns, ltm_* tables and search methods)

## Goal
Replace the legacy recall.py with two scoped recall functions: recall_short_term (Memgraph + stm_*)
and recall_long_term (Neo4j + ltm_*). Update HybridRetriever to support scope routing.

## Delete services/graphknows/graphknows/recall.py
Run `git rm services/graphknows/graphknows/recall.py`

## Read existing files first
- retrieval/retriever.py (HybridRetriever) — understand current structure
- retrieval/stores/memgraph.py — what query methods exist
- retrieval/stores/pgvector.py — what search methods now exist (from STM-001)
- retrieval/stores/neo4j.py — LTM traversal methods

## Create services/graphknows/graphknows/retrieval/recall_stm.py

```python
"""Short-term memory recall: Memgraph entity graph + pgvector stm_chunks/stm_turns."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from graphknows.retrieval.stores.embedder import embed_one

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


def recall_short_term(
    mg_store: Any,
    pg_store: Any,
    query: str,
    session_id: str,
    top_k: int = 5,
) -> list[RecallResult]:
    """Recall from short-term memory: Memgraph entity graph + stm_chunks/stm_turns ANN.

    Strategy:
    1. Embed query
    2. ANN search stm_chunks WHERE session_id = ? → top k_semantic results
    3. ANN search stm_turns WHERE session_id = ? → top k_semantic results
    4. Entity name matching in Memgraph → linked chunks (entity-first path)
    5. RRF fusion of all paths
    6. Return top_k RecallResult sorted by score DESC

    Args:
        mg_store: MemgraphStore with session entity graph.
        pg_store: PgVectorStore with stm_chunks and stm_turns.
        query: Natural-language query.
        session_id: Session namespace to scope all queries.
        top_k: Max results to return.

    Returns:
        List of RecallResult sorted by RRF score descending.
    """
    import asyncio

    words = [w for w in query.lower().split() if len(w) > 3]
    embedding = embed_one(query)
    results: dict[str, tuple[float, RecallResult]] = {}  # id → (score, result)

    # ANN search stm_chunks
    try:
        chunks = asyncio.get_event_loop().run_until_complete(
            pg_store.search_stm_chunks(session_id, embedding, top_k=top_k * 2)
        )
        for rank, chunk in enumerate(chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = chunk["id"]
            if cid not in results or results[cid][0] < rrf:
                results[cid] = (rrf, RecallResult(
                    text=chunk["text"],
                    source_type="chunk",
                    scope="short_term",
                    score=rrf,
                    heading_path=chunk.get("heading_path", ""),
                    provenance={"session_id": session_id, "file_id": chunk.get("file_id", "")},
                ))
    except Exception as exc:
        log.warning("stm_chunks ANN failed: %s", exc)

    # ANN search stm_turns
    try:
        turns = asyncio.get_event_loop().run_until_complete(
            pg_store.search_stm_turns(session_id, embedding, top_k=top_k * 2)
        )
        for rank, turn in enumerate(turns):
            rrf = 1.0 / (RRF_K + rank + 1)
            tid = turn["id"]
            if tid not in results or results[tid][0] < rrf:
                results[tid] = (rrf, RecallResult(
                    text=turn["text"],
                    source_type="turn",
                    scope="short_term",
                    score=rrf,
                    provenance={"session_id": session_id, "role": turn.get("role", "")},
                ))
    except Exception as exc:
        log.warning("stm_turns ANN failed: %s", exc)

    # Entity-first path via Memgraph
    if words:
        try:
            entity_chunks = mg_store.query_entity_chunks(session_id=session_id, words=words, top_k=top_k)
            for rank, ec in enumerate(entity_chunks):
                rrf = 1.0 / (RRF_K + rank + 1)
                cid = ec["chunk_id"]
                if cid in results:
                    # Boost existing result
                    old_score, old_result = results[cid]
                    new_score = old_score + rrf
                    old_result.score = new_score
                    old_result.entity_names = ec.get("entity_names", [])
                    results[cid] = (new_score, old_result)
        except Exception as exc:
            log.warning("Memgraph entity path failed: %s", exc)

    # Sort by score, return top_k
    sorted_results = sorted(results.values(), key=lambda x: x[0], reverse=True)
    return [r for _, r in sorted_results[:top_k]]
```

## Add to MemgraphStore: query_entity_chunks(session_id, words, top_k) -> list[dict]
Cypher:
```
MATCH (e:ENTITY {session_id: $sid})
WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
MATCH (c:CHUNK)-[:HAS_ENTITY]->(e)
WITH c, collect(DISTINCT e.name) AS entity_names, count(DISTINCT e) AS match_count
ORDER BY match_count DESC LIMIT $top_k
RETURN c.id AS chunk_id, c.text AS text, entity_names
```

## Create services/graphknows/graphknows/retrieval/recall_ltm.py

```python
"""Long-term memory recall: Neo4j entity graph + ltm_chunks/ltm_turns/ltm_topics ANN."""
from __future__ import annotations

import logging
from typing import Any

from graphknows.retrieval.recall_stm import RecallResult, RRF_K
from graphknows.retrieval.stores.embedder import embed_one

log = logging.getLogger("graphknows.retrieval.recall_ltm")


def recall_long_term(
    neo_store: Any,
    pg_store: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallResult]:
    """Recall from long-term memory: Neo4j + ltm_chunks/ltm_turns ANN + RRF.

    Args:
        neo_store: Neo4jStore for entity-first graph traversal.
        pg_store: PgVectorStore with ltm_chunks and ltm_turns.
        query: Natural-language query.
        top_k: Max results to return.
        session_filter: Optional session_id to scope LTM results.

    Returns:
        List of RecallResult (scope="long_term") sorted by RRF score descending.
    """
    import asyncio

    embedding = embed_one(query)
    words = [w for w in query.lower().split() if len(w) > 3]
    results: dict[str, tuple[float, RecallResult]] = {}

    # ANN search ltm_chunks
    try:
        chunks = asyncio.get_event_loop().run_until_complete(
            pg_store.search_ltm_chunks(embedding, top_k=top_k * 2, session_id=session_filter)
        )
        for rank, chunk in enumerate(chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = chunk["id"]
            results[cid] = (rrf, RecallResult(
                text=chunk["text"],
                source_type="chunk",
                scope="long_term",
                score=rrf,
                heading_path=chunk.get("heading_path", ""),
                provenance={"session_id": chunk.get("session_id", ""), "file_id": chunk.get("file_id", "")},
            ))
    except Exception as exc:
        log.warning("ltm_chunks ANN failed: %s", exc)

    # Entity-first traversal in Neo4j
    if words:
        try:
            with neo_store._driver.session() as s:
                cypher = """
                    MATCH (e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    MATCH (c:CHUNK)-[:HAS_ENTITY]->(e)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                """
                params: dict = {"words": words, "top_k": top_k * 2}
                if session_filter:
                    cypher = cypher.replace("MATCH (e:ENTITY)", "MATCH (e:ENTITY {session_id: $sid})")
                    params["sid"] = session_filter
                rows = s.run(cypher, **params).data()
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
                        results[cid] = (rrf, RecallResult(
                            text=row["text"],
                            source_type="chunk",
                            scope="long_term",
                            score=rrf,
                            entity_names=row["entities"],
                            provenance={},
                        ))
        except Exception as exc:
            log.warning("Neo4j entity traversal failed: %s", exc)

    sorted_results = sorted(results.values(), key=lambda x: x[0], reverse=True)
    return [r for _, r in sorted_results[:top_k]]
```

## Add to PgVectorStore:
- `async search_ltm_chunks(embedding, top_k, session_id=None) -> list[dict]`
- `async search_ltm_turns(embedding, top_k, session_id=None) -> list[dict]`

## Update HybridRetriever (retrieval/retriever.py)
Add a `scope` parameter: `scope: Literal["short_term", "long_term", "both"] = "both"`
When scope="short_term" → call recall_short_term, skip Neo4j traversal
When scope="long_term" → call recall_long_term, skip Memgraph/stm_* 
When scope="both" → run both and merge by RRF

## Write tests/test_recall_stm.py + tests/test_recall_ltm.py

recall_stm tests (unit, MagicMock):
1. recall_short_term() calls pg_store.search_stm_chunks with correct session_id + embedding
2. recall_short_term() calls pg_store.search_stm_turns with correct session_id
3. recall_short_term() calls mg_store.query_entity_chunks with correct words
4. Entity-found chunk gets boosted score vs chunk-only result
5. ANN failure logs warning and continues (partial results)
6. Results sorted by score DESC
7. top_k respected (returns max top_k items)

recall_ltm tests (unit, MagicMock):
1. recall_long_term() calls pg_store.search_ltm_chunks
2. session_filter passed to ltm_chunks query when provided
3. Neo4j traversal called for entity-first path
4. Results sorted by score DESC
5. scope="long_term" in all RecallResult items

## Acceptance criteria
- All tests pass
- ruff + mypy clean  
- check_imports.py clean
- recall.py deleted from git
