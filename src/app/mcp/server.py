"""
MCP server that wraps the GraphRAG retrieval logic
(`src/app/services/retrieval.py`) exposing it as MCP tools.

The server exposes MCP tools that mirror the behavior of the FastAPI endpoints:
- `kg_chat`  -> POST /chat
- `kg_schema` -> GET /schema
- `kg_health` -> GET /health
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from src.app.infrastructure.config import get_app_config
from src.app.infrastructure.graph_db import get_database_client
from src.app.services.retrieval import (
    Message,
    run_focused_retrieval,
)


mcp = FastMCP("kg-chat-mcp")


@mcp.tool()
async def kg_chat(
    query: str,
    messages: Optional[List[Dict[str, str]]] = None,
    database: str = "falkordb",
    accumulated_graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the focused retrieval pipeline, mirroring the /chat REST endpoint.

    Parameters:
        query: User query text.
        messages: Optional list of conversation messages, each with
            {"role": "user" | "assistant", "content": str}.
        database: Target database type ("falkordb").
        create_plot: Kept for compatibility; currently unused.
        accumulated_graph_data: Optional graph data from previous calls, used
            to accumulate subgraphs across queries.

    Returns:
        A JSON-compatible dict with the same shape as the /chat endpoint:
        {
          "answer": str,
          "context": str,
          "execution_time": float,
          "graph_data": {...},
          "reasoning_chain": [...],
          "cypher_query": str,
          "confidence_score": float
        }
    """
    config = get_app_config()
    db_type = database

    # Convert plain dict messages into backend Message models
    history: List[Message] = []
    if messages:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history.append(Message(role=role, content=content))

    db = get_database_client(config, db_type)
    try:
        # Ensure accumulated_graph_data is a dict if provided
        accumulated_graph: Optional[Dict[str, Any]] = accumulated_graph_data
        if accumulated_graph is not None and not isinstance(accumulated_graph, dict):
            accumulated_graph = None

        result = run_focused_retrieval(
            db,
            query=query,
            messages=history,
            accumulated_graph=accumulated_graph,
        )

        # Match reasoning_chain computation from the FastAPI endpoint
        new_nodes_count = len(result.graph_data.get("nodes", []))
        new_edges_count = len(result.graph_data.get("edges", []))
        if accumulated_graph:
            existing_nodes_count = len(accumulated_graph.get("nodes", []))
            existing_edges_count = len(accumulated_graph.get("edges", []))
            new_nodes_count = new_nodes_count - existing_nodes_count
            new_edges_count = new_edges_count - existing_edges_count

        reasoning_chain = [
            f"Database: {db_type}",
            f"Extracted keywords: {', '.join(result.keywords)}",
            f"Identified seed entities: {', '.join(result.seed_entities)}",
            f"Expanded subgraph: {new_nodes_count} new nodes, {new_edges_count} new edges",
            f"Total accumulated: {len(result.graph_data.get('nodes', []))} nodes, {len(result.graph_data.get('edges', []))} edges",
            f"Context length: {len(result.context)} chars",
            f"Response time: {result.execution_time:.2f}s",
        ]

        return {
            "answer": result.answer,
            "context": result.context,
            "execution_time": result.execution_time,
            "graph_data": result.graph_data,
            "reasoning_chain": reasoning_chain,
            "cypher_query": "Dynamic retrieval based on keywords",
            "confidence_score": 1.0,
        }
    finally:
        db.close()


def _extract_single_column(results: List[Dict[str, Any]], preferred_key: str) -> List[str]:
    """
    Utility to pull a single column out of a list of row dicts.

    FalkorDB returns column names like `label`, `relationshipType`,
    and `propertyKey`, but we defensively fall back to the first value in each row.
    """
    values: List[str] = []
    for row in results:
        if preferred_key in row:
            val = row[preferred_key]
        else:
            # Fallback: take the first column value, if any.
            val = next(iter(row.values()), None)
        if isinstance(val, str):
            values.append(val)
    return values


@mcp.tool()
async def kg_schema(database: str = "falkordb") -> Dict[str, Any]:
    """
    Dynamically inspect the graph schema for the requested database,
    mirroring the /schema REST endpoint.

    Parameters:
        database: Target database type ("falkordb"). Defaults to "falkordb".

    Returns:
        A JSON-compatible dict with schema information:
        {
          "database": str,
          "node_labels": List[str],
          "relationship_types": List[str],
          "property_keys": List[str]
        }
    """
    config = get_app_config()

    try:
        db = get_database_client(config, database)
        try:
            node_labels_raw = db.query("CALL db.labels()")
            rel_types_raw = db.query("CALL db.relationshipTypes()")
            prop_keys_raw = db.query("CALL db.propertyKeys()")

            node_labels = _extract_single_column(node_labels_raw, "label")
            relationship_types = _extract_single_column(rel_types_raw, "relationshipType")
            property_keys = _extract_single_column(prop_keys_raw, "propertyKey")

            return {
                "database": database,
                "node_labels": sorted(set(node_labels)),
                "relationship_types": sorted(set(relationship_types)),
                "property_keys": sorted(set(property_keys)),
            }
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        # On errors, return empty dict to mirror REST behavior
        return {}


@mcp.tool()
async def kg_health() -> Dict[str, str]:
    """
    Lightweight health check, mirroring the /health REST endpoint.
    """
    return {"status": "ok"}


def main() -> None:
    """
    Entry point to run the MCP server over stdio for MCP-compatible clients.
    """
    mcp.run()


if __name__ == "__main__":
    main()


