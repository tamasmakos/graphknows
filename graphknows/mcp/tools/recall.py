"""MCP tools: recall — recall_by_document, recall_by_dialog, recall_by_topic."""

from __future__ import annotations

import asyncio
from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_neo_store, get_pg_store


@app.tool()
def recall_by_document(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find document chunks whose mentioned entities match the query.

    Returns chunks ranked by number of matched entities.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1-20, default 5).

    Returns:
        Dict with a ``results`` key containing matched chunks.
    """
    from graphknows.memory.recall import recall_by_document as _rbd

    top_k = max(1, min(top_k, 20))
    store = get_neo_store()
    try:
        results = _rbd(store._driver, query, top_k)
    finally:
        store.close()

    return {
        "results": [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ]
    }


@app.tool()
async def recall_by_dialog(
    query: str,
    top_k: int = 5,
    session_filter: str = "",
) -> dict[str, Any]:
    """Find past conversation utterances whose mentioned entities match.

    Searches pgvector ``ltm_turns`` (and ``stm_turns`` when a session is
    specified) using ANN cosine similarity.  TURN nodes in the interaction
    graph are populated via :class:`~graphknows.memory.stm.dialog.DialogCapture`
    and flushed to ``ltm_turns`` on ``memory_flush``.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1-20, default 5).
        session_filter: Optional session_id to scope results.

    Returns:
        Dict with a ``results`` key containing matched utterances.
    """
    from graphknows.memory.stores.embedder import embed_one

    top_k = max(1, min(top_k, 20))
    filt = session_filter or None

    embedding = await asyncio.get_running_loop().run_in_executor(None, embed_one, query)

    pg = await get_pg_store()
    rows = await pg.search_ltm_turns(embedding, top_k, session_id=filt)

    results = [
        {
            "text": r["text"],
            "entities": [],
            "provenance": {
                "session_id": r.get("session_id"),
                "dialog_id": r.get("dialog_id"),
                "turn_index": r.get("turn_index"),
                "role": r.get("role"),
            },
            "score": round(float(r.get("cosine_score", 0.0)), 6),
        }
        for r in rows
    ]
    return {"results": results}


@app.tool()
def recall_by_topic(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find topics whose label or member entities match the query.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1-20, default 5).

    Returns:
        Dict with a ``results`` key containing matched topics.
    """
    from graphknows.memory.recall import recall_by_topic as _rbt

    top_k = max(1, min(top_k, 20))
    store = get_neo_store()
    try:
        results = _rbt(store._driver, query, top_k)
    finally:
        store.close()

    return {
        "results": [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ]
    }
