"""MCP tools: ltm — ltm_topics."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_neo_store


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
