"""MCP tools: recall — recall_by_document, recall_by_dialog, recall_by_topic."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_neo_store


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
    from graphknows.recall import recall_by_document as _rbd

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
def recall_by_dialog(
    query: str,
    top_k: int = 5,
    session_filter: str = "",
) -> dict[str, Any]:
    """Find past conversation utterances whose mentioned entities match.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1-20, default 5).
        session_filter: Optional session_id to scope results.

    Returns:
        Dict with a ``results`` key containing matched utterances.
    """
    from graphknows.recall import recall_by_dialog as _rbdia

    top_k = max(1, min(top_k, 20))
    filt = session_filter or None
    store = get_neo_store()
    try:
        results = _rbdia(store._driver, query, top_k, filt)
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
def recall_by_topic(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find topics whose label or member entities match the query.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1-20, default 5).

    Returns:
        Dict with a ``results`` key containing matched topics.
    """
    from graphknows.recall import recall_by_topic as _rbt

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
