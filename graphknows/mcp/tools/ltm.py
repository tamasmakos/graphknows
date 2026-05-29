"""MCP tools: ltm — ltm_query, ltm_entity, ltm_topics."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_neo_store, get_settings


@app.tool()
async def ltm_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """Semantic search over LTM topic summaries.

    Embeds the query and returns the nearest matching topics from the
    pgvector ltm_topics table.

    Args:
        query: Natural-language search query.
        top_k: Number of topics to return (1-20, default 5).

    Returns:
        Dict with a ``results`` key containing a list of matching topics.
    """
    import asyncio
    import json

    import asyncpg as apg

    from graphknows.retrieval.stores.embedder import embed_one

    s = get_settings()
    top_k = max(1, min(top_k, 20))

    # Run CPU-bound embedding in thread pool to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    embedding = json.dumps(await loop.run_in_executor(None, embed_one, query))

    conn = await apg.connect(s.pgvector_dsn)
    try:
        rows = await conn.fetch(
            "SELECT topic_id, summary, entity_names, session_ids FROM ltm_topics "
            "ORDER BY embedding <=> $1::vector LIMIT $2",
            embedding,
            top_k,
        )
    finally:
        await conn.close()

    results = []
    for r in rows:
        entity_names = r["entity_names"]
        if isinstance(entity_names, str):
            entity_names = json.loads(entity_names)
        session_ids = r["session_ids"]
        if isinstance(session_ids, str):
            session_ids = json.loads(session_ids)
        results.append(
            {
                "topic_id": r["topic_id"],
                "summary": r["summary"],
                "entity_names": entity_names or [],
                "session_ids": session_ids or [],
            }
        )
    return {"results": results}


@app.tool()
def ltm_entity(name: str) -> dict[str, Any]:
    """Look up a named entity in the LTM Neo4j graph.

    Returns all properties including community, PageRank, betweenness,
    first/last seen timestamps, and segment count.

    Args:
        name: Entity name (case-insensitive; GLiNER2 stores entities in lowercase).

    Returns:
        Dict with entity properties, or ``{"error": "..."}`` if not found.
    """
    neo = get_neo_store()
    try:
        result = neo.get_entity_by_name(name)
    finally:
        neo.close()

    if result is None:
        return {"error": f"Entity '{name}' not found"}
    return result


@app.tool()
def ltm_topics(min_entities: int = 3) -> dict[str, Any]:
    """List all Topic nodes with an entity count above the threshold.

    Ordered by entity_count descending.

    Args:
        min_entities: Minimum entity count threshold (default 3).

    Returns:
        Dict with a ``topics`` key containing the topic list.
    """
    neo = get_neo_store()
    try:
        topics = neo.list_topic_nodes(min_entities)
    finally:
        neo.close()

    return {"topics": topics}
