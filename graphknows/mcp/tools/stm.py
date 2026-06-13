"""MCP tools: stm — memory_ingest, memory_flush."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import (
    get_neo_store,
    get_pg_store,
    get_settings,
)
from graphknows.memory.stm._async_bridge import _run as _bg_run

log = logging.getLogger("graphknows.mcp.tools.stm")


async def _sync_topics_to_pgvector(neo_store: Any, pg_store: Any) -> int:
    """Read TOPIC nodes from Neo4j and upsert embeddings into pgvector ltm_topics.

    Called after every memory_flush so topic embeddings are available for memory_query.

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
    messages: "str | list[dict]" = "",
    session_id: str = "",
    user_id: str = "",
    agent_id: str = "",
    run_id: str = "",
    title: str = "",
    infer: bool = True,
    memory_class: str = "raw_evidence",
    metadata: "dict[str, Any]" = {},
    text: str = "",
) -> dict[str, Any]:
    """Ingest text or a conversation thread into graphknows short-term memory.

    Accepts a plain string OR a list of message dicts for batch conversation ingestion.
    Scope must include at least one of user_id, agent_id, run_id (or session_id as
    a backward-compat shorthand that maps to run_id).

    When messages is a list, all turns are ingested under a single dialog determined
    by the scope — enabling a full exchange to be stored in one call.

    Set infer=False to skip NER and graph entity extraction (embed-only, faster).

    Note: First call takes ~15s while GLiNER2 model loads.

    Args:
        messages: Plain text string, OR list of {role, content} dicts.
        session_id: Backward-compat shorthand → run_id (ignored if run_id is set).
        user_id: Human participant identifier.
        agent_id: Agent or model identifier.
        run_id: Conversation run / session identifier.
        title: Document title (used when messages is a plain text document).
        infer: Run NER entity extraction (default True).
        memory_class: Semantic class — raw_evidence, summary, fact, episode, state.
        metadata: Extra key/value pairs stored with each record.
        text: Deprecated alias for messages (backward compat).

    Returns:
        Dict with file_id, chunks, entities, elapsed_s, errors.
    """
    # Backward compat: old callers pass text= instead of messages=
    if text and not messages:
        messages = text

    from graphknows.memory.adapter_types import MemoryClass
    from graphknows.memory.models.ingest_options import IngestOptions
    from graphknows.memory.models.message import normalize_messages
    from graphknows.memory.models.scope import MemoryScope

    # Build scope — session_id maps to run_id for backward compat.
    # MemoryScope(run_id=x).to_session_id() returns x when run_id is the only
    # component, so old-style session_id strings are preserved as-is in storage.
    scope = MemoryScope(
        user_id=user_id or None,
        agent_id=agent_id or None,
        run_id=run_id or session_id or None,
    )

    # Dispatch: list[dict] or str with role-like content → dialog path
    #           plain str with title → document/artifact path (chunked)
    is_conversation = isinstance(messages, list) or (
        isinstance(messages, dict) and "role" in messages
    )

    from graphknows.config.builders.stores import build_stm_service

    if is_conversation or (isinstance(messages, str) and not title):
        try:
            mc = MemoryClass(memory_class)
        except ValueError:
            mc = MemoryClass.raw_evidence

        options = IngestOptions(
            scope=scope,
            metadata=metadata,
            infer=infer,
            memory_class=mc,
        )
        msgs = normalize_messages(messages)
        service = build_stm_service(get_settings())
        result = service.ingest_messages(msgs, options)
        return {
            "file_id": result.file_id,
            "chunks": result.chunk_count,
            "entities": result.entity_count,
            "elapsed_s": round(result.elapsed_s, 3),
            "errors": list(result.errors),
        }

    # Document / artifact path (plain text + title) — chunked via _STMIngestHandler
    service = build_stm_service(get_settings())
    result = service.ingest(
        messages,  # type: ignore[arg-type]
        session_id=scope.to_session_id(),
        title=title,
    )
    return {
        "file_id": result.file_id,
        "chunks": result.chunk_count,
        "entities": result.entity_count,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
        "warnings": [],
    }


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
    from graphknows.config.builders.stores import build_promotion_pipeline
    from graphknows.memory.models.promotion import PromotionRequest

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
        log.info("Session %s already in LTM — skipping flush", session_id)
        return {
            "session_id": session_id,
            "status": "already_flushed",
            "nodes": 0,
            "edges": 0,
            "chunks": 0,
            "turns": 0,
            "topics_created": 0,
            "merged_entities": 0,
            "elapsed_s": 0.0,
            "errors": [],
        }

    pipeline = build_promotion_pipeline(s)
    request = PromotionRequest(
        session_id=session_id,
        run_gds=True,
        clear_stm=True,
        merge_entities=True,
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _bg_run(pipeline.promote(request)),
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
        "topics_created": result.topics_created,
        "merged_entities": result.merged_entities,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
    }


async def _ingest_turn_async(mg, pg, extractor, embed_fn, session_id, dialog_id, turn):
    """Helper coroutine that writes a single turn through DialogCapture."""
    from graphknows.memory.stm.dialog import DialogCapture

    t0 = time.perf_counter()
    dc = DialogCapture(mg_store=mg, pg_store=pg, extractor=extractor, embed_fn=embed_fn)
    # Ensure DIALOG node exists
    await mg.write_dialog_node(
        session_id=session_id,
        dialog_id=dialog_id,
        started_at=str(time.time()),
    )
    # write_turn is sync (uses _BG_LOOP internally) — run in executor to avoid blocking MCP loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: dc.write_turn(session_id=session_id, dialog_id=dialog_id, turn=turn),
    )
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
    """Ingest a single conversation turn into graphknows short-term memory.

    Deprecated: use memory_ingest with a list of message dicts instead.
    This tool is kept for backward compatibility and will be removed in a future release.

    Args:
        text: Turn text content (user message, assistant response, etc.).
        role: Speaker role — 'user', 'assistant', or 'system'.
        session_id: Session namespace for scoping.
        dialog_id: Unused (dialog is derived from scope). Kept for compat.
        turn_index: Ordinal position within the dialog (0-based).

    Returns:
        Dict with dialog_id, turn_id, entity_count, elapsed_s, errors.
    """
    from graphknows.memory.models.scope import MemoryScope

    scope = MemoryScope(run_id=session_id)
    msg = {"role": role, "content": text}

    t0 = time.perf_counter()
    result = await memory_ingest(messages=[msg], run_id=session_id, infer=True)

    # Translate response to the legacy format callers expect
    turns = result.get("turns", result.get("chunks", 0))
    d_id = result.get("dialog_id", scope.to_dialog_id())
    return {
        "dialog_id": d_id,
        "turn_id": "",  # not tracked individually in the new path
        "entity_count": result.get("entities", 0),
        "elapsed_s": result.get("elapsed_s", round(time.perf_counter() - t0, 3)),
        "errors": result.get("errors", []),
    }
