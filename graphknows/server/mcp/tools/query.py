"""MCP tool: memory_query — single unified retrieval entry point."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._bounded import bounded
from graphknows.server.mcp._state import get_runtime


@app.tool()
@bounded
async def memory_query(
    query: str,
    session_id: str = "",
    top_k: int = 10,
    scope: str = "both",
) -> dict[str, Any]:
    """Retrieve the most relevant memories for a query.

    A thin adapter over :meth:`graphknows.Memory.recall_memory` — the same call
    an in-process caller makes, so this transport cannot drift from it. Driving
    the retriever directly here is what left the MCP hit shape without a speaker
    or a timestamp, capped ``top_k`` where the facade had deliberately removed
    its ceiling, and gave ``scope="stm"`` a different meaning over MCP than in
    process.

    Args:
        query: Natural-language question or search phrase.
        session_id: Session to scope the search to.
        top_k: Maximum hits to return, up to 100 (FR-030); a higher value is
            rejected rather than clamped.
        scope: Which memory tier to search — ``"stm"`` (this session's buffered,
            unflushed turns), ``"ltm"`` (the graph), or ``"both"`` (default).

    Returns:
        Dict with a ``hits`` list — each hit the wire form of
        :class:`~graphknows.models.hit.Hit` (``text``, ``speaker``, ``ts``,
        ``score``, ``sources``, ``session_id``, ``chunk_id``, ``doc_id``,
        ``entities``, ``metadata``) — plus ``total``, ``query``, ``scope`` and
        the structured ``facts`` sheet.
    """
    runtime = await get_runtime()
    result = await runtime.recall_memory(query, session_id=session_id, top_k=top_k, scope=scope)
    hits = result.get("hits", [])
    return {
        "hits": [h.to_dict() for h in hits],
        "total": len(hits),
        "query": query,
        # What the facade actually searched, as it reports it — an unknown
        # scope falls back there, and duplicating that rule here is how the two
        # came to disagree in the first place.
        "scope": result.get("scope", scope),
        "facts": list(result.get("facts", []) or []),
    }
