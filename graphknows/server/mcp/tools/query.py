"""MCP tool: memory_query — single unified retrieval entry point."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_or_init_retriever


@app.tool()
async def memory_query(
    query: str,
    session_id: str = "",
    top_k: int = 10,
    scope: str = "both",
    namespace: str = "",
    strategy: str = "",
) -> dict[str, Any]:
    """Retrieve the most relevant memory chunks for a query.

    Searches short-term memory (STM), long-term memory (LTM), or both,
    using dense vector search with optional entity-graph boosting when
    the LTM entity index is populated.

    Args:
        query: Natural-language question or search phrase.
        session_id: Session namespace for STM scoping (used when scope
            includes ``"stm"``).
        top_k: Maximum hits to return (1-100, default 10).
        scope: Which memory tier to search — ``"stm"`` (session chunks),
            ``"ltm"`` (promoted long-term chunks), or ``"both"`` (default).
        namespace: Tenant isolation boundary (default: the server's
            ``GRAPHKNOWS_NAMESPACE``).
        strategy: Retrieval strategy — ``"det"`` (deterministic channels +
            RRF) or ``"agentic"`` (ReAct loop over those channels). Empty
            uses ``GRAPHKNOWS_RETRIEVAL``. Independent of how the graph was
            ingested, so either strategy works over any namespace.

    Returns:
        Dict with a ``hits`` list, each hit containing ``text``, ``score``,
        ``chunk_id``, ``doc_id``, ``sources``, ``entities``, and ``metadata``
        (resolved temporal dates, memory_class, community).
    """
    top_k = max(1, min(top_k, 100))
    if scope not in ("stm", "ltm", "both"):
        scope = "both"
    if strategy not in ("", "det", "agentic"):
        strategy = ""

    retriever = await get_or_init_retriever(namespace, strategy)
    ctx = await retriever.retrieve(query, session_id, top_k=top_k, scope=scope)

    hits = [
        {
            "text": fc["text"],
            "score": fc["rrf_score"],
            "chunk_id": fc["chunk_id"],
            "doc_id": fc["doc_id"],
            "sources": fc.get("sources", ""),
            "entities": fc.get("entities", []),
            "metadata": fc.get("metadata", {}),
        }
        for fc in ctx.fused_results
    ]
    facts = list(getattr(ctx, "facts", []) or [])
    return {
        "hits": hits,
        "total": len(hits),
        "query": query,
        "scope": scope,
        "facts": facts,
        "search_type": getattr(ctx, "search_type", ""),
    }
