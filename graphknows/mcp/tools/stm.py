"""MCP tools: stm — memory_ingest, memory_recall, memory_flush."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_mg_store, get_neo_store, get_pg_store, get_settings


def _normalise_and_weight(hits: list[Any], weight: float) -> None:
    """Min-max normalise scores in-place, then apply weight."""
    if not hits:
        return
    scores = [h.score for h in hits]
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    for h in hits:
        h.score = weight * ((h.score - mn) / rng if rng else 1.0)


@app.tool()
async def memory_ingest(
    text: str,
    session_id: str,
    title: str = "",
) -> dict[str, Any]:
    """Ingest text into graphknows short-term memory (Memgraph + pgvector STM).

    GLiNER2 entity extraction runs on the text; extracted entities are stored
    as nodes in Memgraph.  The text is chunked and embedded into pgvector stm_*.

    Note: First call takes ~15s while GLiNER2 model loads.

    Args:
        text: The text content to ingest.
        session_id: Session namespace for the STM graph.
        title: Optional document title for provenance.

    Returns:
        Dict with file_id, chunks, entities, elapsed_s, errors.
    """
    from graphknows.builders.stores import build_memory_ingestor

    ingestor = build_memory_ingestor(get_settings())
    result = ingestor.ingest_text(text, session_id=session_id, title=title)

    return {
        "file_id": result.file_id,
        "chunks": result.chunk_count,
        "entities": result.entity_count,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
        "warnings": [],
    }


@app.tool()
async def memory_recall(
    query: str,
    session_id: str,
    top_k: int = 5,
    scope: str = "stm",
) -> dict[str, Any]:
    """Recall from graphknows memory using hybrid RRF search.

    Args:
        query: Natural language query to recall against.
        session_id: Session namespace to scope the search.
        top_k: Maximum number of hits to return (default 5).
        scope: ``"stm"`` for short-term, ``"ltm"`` for long-term,
            ``"both"`` for RRF-merged results from both.

    Returns:
        Dict with ``hits`` list, each item having score, type, text,
        session_id, and entities fields.
    """
    import re

    from graphknows.memory.stm.recall import recall_short_term
    from graphknows.retrieval.stores.embedder import embed_one

    embed = embed_one

    def _clean(text: str) -> str:
        r"""Strip the '## Content\n' heading artifact inserted by the formatter."""
        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _fmt(results: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "score": round(r.score, 6),
                "cosine_score": round(getattr(r, "cosine_score", r.score), 6),
                "type": r.source_type,
                "text": _clean(r.text),
                "session_id": r.provenance.get("session_id", session_id),
                "entities": list(r.entity_names or []),
            }
            for r in results
        ]

    if scope in ("stm", "both"):
        mg = get_mg_store()
        pg = await get_pg_store()
        stm_hits = recall_short_term(
            mg, pg, query, session_id=session_id, top_k=top_k, embed_fn=embed
        )
    else:
        stm_hits = []

    if scope in ("ltm", "both"):
        from graphknows.retrieval.recall_ltm import recall_long_term

        pg2 = await get_pg_store()
        neo = get_neo_store()
        ltm_hits = recall_long_term(
            neo, pg2, query, top_k=top_k, session_filter=None, embed_fn=embed
        )
        neo.close()
    else:
        ltm_hits = []

    if scope == "both":
        _normalise_and_weight(stm_hits, weight=0.6)
        _normalise_and_weight(ltm_hits, weight=0.4)

    all_hits = stm_hits + ltm_hits
    all_hits.sort(key=lambda r: r.score, reverse=True)

    return {"hits": _fmt(all_hits[:top_k])}


@app.tool()
async def memory_flush(session_id: str, run_gds: bool = False) -> dict[str, Any]:
    """Flush a session from short-term memory into long-term memory.

    Promotes Memgraph STM nodes → Neo4j LTM with entity merging.
    Louvain topic summarization and GDS enrichment are skipped by default;
    set ``run_gds=True`` for full enrichment (Node2Vec, Leiden, PageRank, etc.).

    Args:
        session_id: Session to promote from STM to LTM.
        run_gds: Whether to run GDS enrichment after the flush.

    Returns:
        Dict with nodes, edges, chunks, turns, elapsed_s, errors.
    """
    import logging

    from graphknows.builders.stores import build_session_flusher
    from graphknows.memory.flush import FlushOptions

    logger = logging.getLogger("graphknows.mcp.tools.stm")
    s = get_settings()

    # Idempotency guard
    neo_check = get_neo_store()
    with neo_check._driver.session(database=neo_check._db) as neo_sess:
        rows = list(
            neo_sess.run(
                "MATCH (s:SESSION {id: $sid}) RETURN count(s) AS n",
                sid=session_id,
            )
        )
    if rows and rows[0]["n"] > 0:
        logger.info("Session %s already in LTM — skipping flush", session_id)
        return {
            "session_id": session_id,
            "status": "already_flushed",
            "nodes_written": 0,
            "edges_written": 0,
            "chunks_promoted": 0,
            "turns_promoted": 0,
            "topics_created": 0,
            "merged_entities": 0,
            "elapsed_s": 0.0,
            "errors": [],
        }

    flusher = build_session_flusher(s)
    opts = FlushOptions(
        summarize_topics=False,  # Never block MCP flush on LLM call; pipeline must stay <60s
        run_gds=run_gds,
        clear_stm=True,
        merge_entities=True,
    )

    result = await flusher.flush(session_id, opts)

    return {
        "nodes": result.nodes_written,
        "edges": result.edges_written,
        "chunks": result.chunks_promoted,
        "turns": result.turns_promoted,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
    }
