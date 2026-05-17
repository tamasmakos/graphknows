"""GraphRAG FastMCP server — exposes retrieval tools to any MCP client.

This module creates a FastMCP server that wraps the HybridRetriever with 5 tools:
  - retrieve_hybrid    — dense vector + graph 2-hop + RRF fusion
  - get_graph_path     — shortest path between two entities in Neo4j
  - get_entity_chunks  — source text chunks for an entity
  - get_community      — Leiden community neighbourhood for an entity
  - ingest_chunks      — push text chunks into pgvector (session-scoped)

Mount alongside FastAPI::

    app.mount("/mcp", mcp_app)

Then connect any MCP client to ``http://host:8010/mcp/``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("graphrag.mcp")

# Shared singleton — set during FastAPI lifespan startup
_retriever: Any = None


def set_retriever(retriever: Any) -> None:
    """Register the HybridRetriever instance to be used by MCP tools.

    Call this from the FastAPI lifespan after the retriever is initialised.
    """
    global _retriever
    _retriever = retriever


def get_retriever() -> Any:
    """Return the registered retriever or raise RuntimeError."""
    if _retriever is None:
        raise RuntimeError("GraphRAG retriever not yet initialised")
    return _retriever


# ---------------------------------------------------------------------------
# FastMCP server definition
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "graphrag",
    instructions=(
        "GraphRAG retrieval tools for Graphknows. "
        "Use retrieve_hybrid for general queries, get_graph_path for entity relationships, "
        "get_entity_chunks for source evidence, and get_community for neighbourhood analysis."
    ),
)


@mcp.tool()
async def retrieve_hybrid(
    query: str,
    session_id: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """Hybrid retrieval: combines dense vector search (pgvector) with 2-hop graph
    traversal (Neo4j) and fuses results using Reciprocal Rank Fusion (RRF).

    Args:
        query: Natural-language query to retrieve relevant chunks and entities.
        session_id: Optional session namespace for scoped pgvector search.
        top_k: Number of results to return (default 10, max 50).

    Returns:
        Dict with 'chunks' (list of text chunks), 'entities' (list of matched
        entity names), and 'scores' (RRF scores for each chunk).
    """
    retriever = get_retriever()
    top_k = min(top_k, 50)
    try:
        result: dict = await asyncio.to_thread(retriever.retrieve, query, top_k)
        return {
            "chunks": result.get("chunks", []),
            "entities": result.get("entities", []),
            "scores": result.get("scores", []),
        }
    except Exception as exc:
        log.exception("retrieve_hybrid failed for query %r", query)
        return {"error": str(exc), "chunks": [], "entities": [], "scores": []}


@mcp.tool()
async def get_graph_path(
    entity_a: str,
    entity_b: str,
    max_hops: int = 3,
) -> dict[str, Any]:
    """Find the shortest path between two entities in the Neo4j knowledge graph.

    Args:
        entity_a: Source entity name or ID.
        entity_b: Target entity name or ID.
        max_hops: Maximum number of relationship hops (default 3, max 6).

    Returns:
        Dict with 'path' (list of entity names along the path), 'found' (bool),
        and 'length' (number of hops).
    """
    retriever = get_retriever()
    max_hops = min(max_hops, 6)
    try:
        result: dict = await asyncio.to_thread(
            retriever.get_path, entity_a, entity_b, max_hops
        )
        path = result.get("path", [])
        return {
            "path": path,
            "found": result.get("found", bool(path)),
            "length": len(path) - 1 if len(path) > 1 else 0,
        }
    except Exception as exc:
        log.exception("get_graph_path failed for %r -> %r", entity_a, entity_b)
        return {"error": str(exc), "path": [], "found": False, "length": 0}


@mcp.tool()
async def get_entity_chunks(
    entity: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve the source text chunks that mention a specific entity.

    Args:
        entity: Entity name or ID to look up.
        top_k: Number of source chunks to return (default 5, max 20).

    Returns:
        Dict with 'chunks' (list of text strings) and 'entity' (the queried name).
    """
    retriever = get_retriever()
    top_k = min(top_k, 20)
    try:
        result: list = await asyncio.to_thread(retriever.get_chunks, entity, top_k)
        return {"entity": entity, "chunks": result}
    except Exception as exc:
        log.exception("get_entity_chunks failed for entity %r", entity)
        return {"error": str(exc), "entity": entity, "chunks": []}


@mcp.tool()
async def get_community(
    entity_id: str,
) -> dict[str, Any]:
    """Get the Leiden community cluster that an entity belongs to.

    Returns the community label, ID, and all member entities in the same cluster.
    Useful for understanding which actors, organizations, or concepts are
    structurally grouped together in the knowledge graph.

    Args:
        entity_id: Entity name or Neo4j ID to look up.

    Returns:
        Dict with 'community_id', 'label', and 'members' (list of entity names).
    """
    retriever = get_retriever()
    try:
        result: dict = await asyncio.to_thread(retriever.get_community, entity_id)
        return {
            "community_id": result.get("community_id"),
            "label": result.get("label"),
            "members": result.get("members", []),
        }
    except Exception as exc:
        log.exception("get_community failed for entity %r", entity_id)
        return {"error": str(exc), "community_id": None, "label": None, "members": []}


@mcp.tool()
async def ingest_chunks(
    chunks: list[dict],
    session_id: str,
) -> dict[str, Any]:
    """Ingest text chunks into pgvector for a given session namespace.

    Each chunk dict must have at least 'text' and 'id' keys. Optionally
    include 'doc_id', 'metadata', and 'embedding' (computed automatically
    if missing).

    Args:
        chunks: List of chunk dicts to ingest.
        session_id: Session namespace for scoped retrieval.

    Returns:
        Dict with 'ingested' (count of successfully stored chunks).
    """
    retriever = get_retriever()
    if not chunks or not session_id:
        return {"ingested": 0, "error": "chunks and session_id are required"}
    try:
        # Use retriever's underlying pg_store if available
        pg_store = getattr(retriever, "pg_store", None) or getattr(retriever, "_pg", None)
        if pg_store is None:
            return {"ingested": 0, "error": "PgStore not available on retriever"}
        ingested = await asyncio.to_thread(pg_store.upsert_chunks, chunks, session_id)
        return {"ingested": ingested if isinstance(ingested, int) else len(chunks)}
    except Exception as exc:
        log.exception("ingest_chunks failed for session %r", session_id)
        return {"error": str(exc), "ingested": 0}


def build_mcp_app() -> Any:
    """Return the MCP Starlette ASGI app for mounting under FastAPI.

    Returns:
        Starlette application exposing streamable HTTP transport at ``/mcp/``.
    """
    return mcp.streamable_http_app()
