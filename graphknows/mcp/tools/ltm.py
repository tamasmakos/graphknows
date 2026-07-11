"""MCP tools for long-term memory exploration."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
async def ltm_topics(min_entities: int = 3, namespace: str = "") -> dict[str, Any]:
    """List Topic nodes with an entity count above the threshold."""
    runtime = await get_runtime(namespace)
    return await runtime.ltm_topics(min_entities=min_entities)


@app.tool()
async def ltm_entity(name: str, namespace: str = "") -> dict[str, Any]:
    """Return properties for one LTM entity."""
    runtime = await get_runtime(namespace)
    return await runtime.ltm_entity(name=name)


@app.tool()
async def ltm_query(query: str, top_k: int = 5, namespace: str = "") -> dict[str, Any]:
    """Search LTM topics by query text and member entities."""
    runtime = await get_runtime(namespace)
    return runtime.ltm_query(query=query, top_k=top_k)


@app.tool()
async def ltm_entities(
    session_id: str = "", limit: int = 200, namespace: str = ""
) -> dict[str, Any]:
    """List ENTITY nodes from the LTM graph, optionally filtered to one session.

    Returns entity name, type, pagerank, community_id, and confidence for each
    node — useful for auditing extraction quality across pipeline modes.
    """
    runtime = await get_runtime(namespace)
    return await runtime.ltm_entities(session_id=session_id, limit=limit)
