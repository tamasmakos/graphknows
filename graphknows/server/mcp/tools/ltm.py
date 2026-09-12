"""MCP tools for long-term memory exploration and forgetting."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._bounded import bounded
from graphknows.server.mcp._state import get_runtime


@app.tool()
async def ltm_entity(name: str) -> dict[str, Any]:
    """Return properties for one LTM entity."""
    runtime = await get_runtime()
    return await runtime.ltm_entity(name=name)


@app.tool()
@bounded
async def ltm_entities(session_id: str = "", limit: int = 200) -> dict[str, Any]:
    """List ENTITY nodes from the LTM graph, optionally filtered to one session.

    Returns entity name, type, pagerank, community_id, and confidence for each
    node — useful for auditing extraction quality across pipeline modes.
    """
    runtime = await get_runtime()
    return await runtime.ltm_entities(session_id=session_id, limit=limit)


@app.tool()
@bounded
async def memory_forget(record_id: str) -> dict[str, Any]:
    """Tombstone one fact, entity or segment so recall stops seeing it (FR-037).

    Never a delete: merge logs and provenance stay replayable. Returns the
    counters, including how many facts the cascade tombstoned with a segment.
    """
    runtime = await get_runtime()
    counters = await runtime.forget(record_id)
    return counters.model_dump()
