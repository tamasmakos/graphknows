"""graphknows FastMCP server — retrieve and community_summary tools.

Tools
-----
retrieve          — entity-first hybrid retrieval → XML string
community_summary — Leiden community summary for a named entity
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("graphknows.mcp")

# Shared singletons set during FastAPI lifespan
_retriever: Any = None
_pg: Any = None
_neo: Any = None
_cfg: dict = {}


def set_dependencies(retriever: Any, pg: Any, neo: Any, cfg: dict) -> None:
    """Register shared store instances for MCP tools.

    Called from FastAPI lifespan after all stores are initialised.
    """
    global _retriever, _pg, _neo, _cfg
    _retriever = retriever
    _pg = pg
    _neo = neo
    _cfg = cfg


def _require_retriever() -> Any:
    if _retriever is None:
        raise RuntimeError("graphknows retriever not yet initialised")
    return _retriever


mcp = FastMCP(
    "graphknows",
    instructions=(
        "graphknows agentic memory tools. "
        "Use `retrieve` for all research queries — it returns an XML document "
        "with graph-matched entities, semantic chunks, and RRF-fused results. "
        "Use `community_summary` to discover thematic entity clusters."
    ),
)


# ---------------------------------------------------------------------------
# Tool: retrieve
# ---------------------------------------------------------------------------


@mcp.tool()
async def retrieve(
    query: str,
    session_id: str,
    k_entity: int = 10,
    k_semantic: int = 20,
    k_final: int = 10,
) -> str:
    """Hybrid entity-first retrieval combining graph traversal and vector search.

    Performs:
      1. Cosine ANN on entity embeddings → top entities.
      2. MENTIONS traversal in Neo4j → linked chunks per entity.
      3. Cosine ANN on session-scoped chunk embeddings.
      4. RRF fusion of both paths.

    Returns an XML string conforming to the <retrieval_context> schema.

    Args:
        query: Natural-language research query.
        session_id: Active session namespace for chunk retrieval.
        k_entity: Number of entities to retrieve (default 10).
        k_semantic: Number of semantic chunks to retrieve (default 20).
        k_final: Maximum fused results to return (default 10).
    """
    ret = _require_retriever()
    ctx = await ret.retrieve(
        query,
        session_id,
        k_entity=k_entity,
        k_semantic=k_semantic,
        k_final=k_final,
    )
    return ctx.to_xml()


# ---------------------------------------------------------------------------
# Tool: community_summary
# ---------------------------------------------------------------------------


@mcp.tool()
async def community_summary(entity_name: str) -> str:
    """Return the Leiden community topic summary for a named entity.

    Looks up the Topic node linked to *entity_name* via IN_COMMUNITY edges
    in Neo4j and returns the LLM-generated topic summary.

    Args:
        entity_name: Entity name to look up (case-insensitive).

    Returns:
        Topic summary string, or a message indicating no community was found.
    """
    if _neo is None:
        raise RuntimeError("Neo4j store not yet initialised")
    topic = await _neo.get_entity_community(entity_name)
    if topic is None:
        return f"No community found for entity: {entity_name}"
    return f"{topic['label']}\n\n{topic['summary']}"
