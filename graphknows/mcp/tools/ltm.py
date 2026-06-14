"""MCP tools for long-term memory exploration."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
def ltm_topics(min_entities: int = 3) -> dict[str, Any]:
    """List Topic nodes with an entity count above the threshold."""
    return get_runtime().ltm_topics(min_entities=min_entities)


@app.tool()
def ltm_entity(name: str) -> dict[str, Any]:
    """Return properties for one LTM entity."""
    return get_runtime().ltm_entity(name=name)


@app.tool()
def ltm_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search LTM topics by query text and member entities."""
    return get_runtime().ltm_query(query=query, top_k=top_k)


@app.tool()
def recall_by_document(query: str, top_k: int = 5) -> dict[str, Any]:
    """Recall document chunks ranked by matched entities."""
    return get_runtime().recall_by_document(query=query, top_k=top_k)


@app.tool()
def recall_by_dialog(
    query: str,
    top_k: int = 5,
    session_id: str = "",
) -> dict[str, Any]:
    """Recall dialog turns ranked by matched entities."""
    return get_runtime().recall_by_dialog(query=query, top_k=top_k, session_id=session_id)


@app.tool()
def recall_by_topic(query: str, top_k: int = 5) -> dict[str, Any]:
    """Recall topics ranked by matched entities."""
    return get_runtime().recall_by_topic(query=query, top_k=top_k)
