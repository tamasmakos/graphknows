"""MCP tools: stm — memory_ingest, memory_recall, memory_flush."""

from __future__ import annotations

import asyncio
from typing import Any

from graphknows.config.builders.stores import build_pgvector_store
from graphknows.mcp._app import app
from graphknows.mcp._state import (
    get_mg_store,
    get_neo_store,
    get_pg_store,
    get_pg_store_bridge,
    get_settings,
)
from graphknows.memory.stm._async_bridge import _run as _bg_run


async def _sync_topics_to_pgvector(neo_store: Any, pg_store: Any) -> int:
    """Read TOPIC nodes from Neo4j and upsert embeddings into pgvector ltm_topics.

    Called after every memory_flush so ltm_query works without run_gds=True.

    Args:
        neo_store: Open Neo4jStore instance (caller must close).
        pg_store: Connected PgVectorStore instance.

    Returns:
        Number of topics synced.
    """
    from graphknows.memory.stores.embedder import embed_one

    # Read existing TOPIC nodes from Neo4j
    with neo_store._driver.session(database=neo_store._db) as s:
        rows = s.run(
            "MATCH (t:TOPIC) "
            "OPTIONAL MATCH (e:ENTITY)-[:IN_TOPIC]->(t) "
            "RETURN t.id AS topic_id, t.summary AS summary, t.community_id AS comm_id, "
            "collect(e.name) AS members"
        ).data()

    if not rows:
        return 0

    synced = 0
    loop = asyncio.get_running_loop()
    for row in rows:
        topic_id = row.get("topic_id") or f"TOPIC_{row.get('comm_id', synced)}"
        summary = (row.get("summary") or "").strip()
        members = row.get("members") or []
        text = summary or topic_id

        embedding = await loop.run_in_executor(None, embed_one, text)

        topic = {
            "topic_id": topic_id,
            "summary": summary,
            "embedding": embedding,
            "entity_names": [m for m in members if m],
            "session_ids": [],
        }
        await pg_store.write_ltm_topic(topic)
        synced += 1

    return synced


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
    from graphknows.config.builders.stores import build_memory_ingestor

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
    from graphknows.memory.stores.embedder import embed_one

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
        pg = build_pgvector_store(get_settings())
        _bg_run(pg.connect())  # pool created on _BG_LOOP where recall_short_term will use it
        stm_hits = recall_short_term(
            mg, pg, query, session_id=session_id, top_k=top_k, embed_fn=embed
        )
    else:
        stm_hits = []

    if scope in ("ltm", "both"):
        from graphknows.memory.query.recall_ltm import recall_long_term

        pg2 = get_pg_store_bridge()
        neo = get_neo_store()
        ltm_hits = recall_long_term(
            neo, pg2, query, top_k=top_k, session_filter=session_id, embed_fn=embed
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
async def memory_flush(session_id: str) -> dict[str, Any]:
    """Flush a session from short-term memory into long-term memory.

    Promotes Memgraph STM nodes → Neo4j LTM with entity merging.
    Topic summarization and GDS enrichment (Node2Vec, Louvain, PageRank,
    betweenness) always run.

    Args:
        session_id: Session to promote from STM to LTM.

    Returns:
        Dict with nodes, edges, chunks, turns, elapsed_s, errors.
    """
    import logging

    from graphknows.config.builders.stores import build_session_flusher
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
        summarize_topics=True,
        run_gds=True,
        clear_stm=True,
        merge_entities=True,
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _bg_run(flusher.flush(session_id, opts)),
    )

    # Sync all Neo4j TOPIC nodes to pgvector ltm_topics (embed summary → ANN index).
    # This runs outside the 60s flush window — non-blocking, best-effort.
    try:
        pg = await get_pg_store()
        neo_sync = get_neo_store()
        await _sync_topics_to_pgvector(neo_sync, pg)
        neo_sync.close()
    except Exception as _topic_exc:
        result.errors.append(f"topic_sync: {_topic_exc}")

    return {
        "nodes": result.nodes_written,
        "edges": result.edges_written,
        "chunks": result.chunks_promoted,
        "turns": result.turns_promoted,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
    }


async def _ingest_turn_async(mg, pg, extractor, embed_fn, session_id, dialog_id, turn):
    """Helper coroutine that writes a single turn through DialogCapture."""
    import time

    from graphknows.memory.stm.dialog import DialogCapture

    t0 = time.perf_counter()
    dc = DialogCapture(mg_store=mg, pg_store=pg, extractor=extractor, embed_fn=embed_fn)
    # Ensure DIALOG node exists
    await mg.write_dialog_node(
        session_id=session_id,
        dialog_id=dialog_id,
        started_at=str(time.time()),
    )
    result = dc.write_turn(session_id=session_id, dialog_id=dialog_id, turn=turn)
    elapsed = time.perf_counter() - t0
    return {
        "dialog_id": result.dialog_id,
        "turn_id": result.turn_id,
        "entity_count": result.entity_count,
        "elapsed_s": round(elapsed, 3),
        "errors": result.errors,
    }


@app.tool()
async def memory_ingest_turn(
    text: str,
    role: str,
    session_id: str,
    dialog_id: str = "",
    turn_index: int = 0,
) -> dict:
    """Ingest a conversation turn into graphknows short-term memory.

    Writes a DIALOG → TURN node in Memgraph, extracts entities with GLiNER2,
    creates TURN→ENTITY MENTIONS edges, and embeds turn text into pgvector stm_turns.

    Args:
        text: Turn text content (user message, assistant response, etc.).
        role: Speaker role — 'user', 'assistant', or 'system'.
        session_id: Session namespace for scoping.
        dialog_id: Dialog identifier. Auto-generated as 'dialog-<session_id>' if empty.
        turn_index: Ordinal position within the dialog (0-based).

    Returns:
        Dict with dialog_id, turn_id, entity_count, elapsed_s, errors.
    """
    import time
    import uuid

    from graphknows.config.builders.stores import build_pgvector_store
    from graphknows.memory.extractor import GLiNER2EntityExtractor
    from graphknows.memory.stm._async_bridge import _run as _bg_run
    from graphknows.memory.stm.dialog import Turn
    from graphknows.memory.stores.embedder import embed_one

    s = get_settings()
    mg = get_mg_store()
    pg = build_pgvector_store(s)
    _bg_run(pg.connect())

    if not dialog_id:
        dialog_id = f"dialog-{session_id}"

    turn_id = str(uuid.uuid4())
    turn = Turn(
        id=turn_id,
        index=turn_index,
        role=role,
        text=text,
        timestamp=time.time(),
    )

    extractor = GLiNER2EntityExtractor()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _bg_run(
            _ingest_turn_async(mg, pg, extractor, embed_one, session_id, dialog_id, turn)
        ),
    )
    return result
