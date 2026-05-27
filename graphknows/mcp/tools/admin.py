"""MCP tools: admin — memory_doctor, memory_stats."""

from __future__ import annotations

import os
from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_settings


@app.tool()
async def memory_doctor() -> dict[str, Any]:
    """Check graphknows backend connectivity (Memgraph, Neo4j, pgvector).

    Returns a dict with boolean values for each backend, plus version and pid.
    """
    from graphknows.config.health import check_backends

    result = await check_backends(get_settings())
    result["pid"] = os.getpid()
    return result


@app.tool()
async def memory_stats(session_id: str = "") -> dict[str, Any]:
    """Return graphknows memory statistics (STM and LTM counts).

    Args:
        session_id: Optional session to filter STM stats by.  Empty string
            returns aggregate counts.

    Returns:
        Dict with ``stm`` and ``ltm`` sub-dicts containing row counts.
    """
    from graphknows.config.health import get_memory_stats

    return await get_memory_stats(get_settings(), session_id=session_id)
