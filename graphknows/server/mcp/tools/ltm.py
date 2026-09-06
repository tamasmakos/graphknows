"""MCP tools for long-term memory exploration."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_runtime


@app.tool()
async def ltm_topics(min_entities: int = 3) -> dict[str, Any]:
    """List Topic nodes with an entity count above the threshold."""
    runtime = await get_runtime()
    return await runtime.ltm_topics(min_entities=min_entities)


@app.tool()
async def ltm_entity(name: str) -> dict[str, Any]:
    """Return properties for one LTM entity."""
    runtime = await get_runtime()
    return await runtime.ltm_entity(name=name)


@app.tool()
async def ltm_entities(session_id: str = "", limit: int = 200) -> dict[str, Any]:
    """List ENTITY nodes from the LTM graph, optionally filtered to one session.

    Returns entity name, type, pagerank, community_id, and confidence for each
    node — useful for auditing extraction quality across pipeline modes.
    """
    runtime = await get_runtime()
    return await runtime.ltm_entities(session_id=session_id, limit=limit)
