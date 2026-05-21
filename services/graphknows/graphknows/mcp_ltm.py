"""MCP Server — GraphKnows LTM (Long-Term Memory) read tools.

Offers semantic topic search, entity lookup, and topic listing over
the graphknows LTM knowledge graph.  Designed as a stdio MCP server
following the same pattern as services/search/mcp_server.py.

Tools
-----
  ltm_query    Semantic search over LTM topics
  ltm_entity   Look up a named entity in the LTM graph
  ltm_topics   List all topics meeting a minimum entity threshold

Run via
-------
    python -m graphknows.mcp_ltm
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import mcp.types as types
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from neo4j import GraphDatabase

load_dotenv(Path(os.environ.get("PWD", ".")) / ".env")

log = logging.getLogger(__name__)

server = Server("graphknows-ltm")


def _neo4j_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, pw))


def _pg_dsn():
    return os.environ.get("PGVECTOR_DSN", "postgresql://postgres@172.18.0.1:5432/vectordb")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ltm_query",
            description=(
                "Semantic search over LTM topic summaries.  Embeds the query and "
                "returns the nearest matching topics from the pgvector ltm_topics table."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of topics to return (1–20)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ltm_entity",
            description=(
                "Look up a named entity in the LTM Neo4j graph and return all "
                "properties including community, PageRank, betweenness, first/last "
                "seen timestamps, and segment count."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name (case-sensitive)",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="ltm_topics",
            description=(
                "List all Topic nodes in the LTM graph with an entity count "
                "above the specified threshold, ordered by entity_count descending."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_entities": {
                        "type": "integer",
                        "default": 3,
                        "description": "Minimum entity count threshold",
                    },
                },
            },
        ),
        types.Tool(
            name="recall_by_document",
            description=(
                "Find document chunks whose mentioned entities match the query. "
                "Returns chunks ranked by number of matched entities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max results to return (1–20)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="recall_by_dialog",
            description=(
                "Find past conversation utterances whose mentioned entities "
                "match the query. Returns utterances with session context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max results to return (1–20)",
                    },
                    "session_filter": {
                        "type": "string",
                        "default": "",
                        "description": "Optional session_id to scope results",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="recall_by_topic",
            description=(
                "Find topics whose label or member entities match the query. "
                "Returns topics ranked by relevance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max results to return (1–20)",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = await _dispatch(name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as exc:
        log.exception("[mcp-ltm] tool %s failed", name)
        return [types.TextContent(type="text", text=f"Error: {exc}")]


async def _dispatch(name: str, args: dict) -> str:
    if name == "ltm_query":
        return await _tool_ltm_query(args)
    if name == "ltm_entity":
        return await _tool_ltm_entity(args)
    if name == "ltm_topics":
        return await _tool_ltm_topics(args)
    if name == "recall_by_document":
        return await _tool_recall_by_document(args)
    if name == "recall_by_dialog":
        return await _tool_recall_by_dialog(args)
    if name == "recall_by_topic":
        return await _tool_recall_by_topic(args)
    raise ValueError(f"Unknown tool: {name}")


# ── Tool implementations ──────────────────────────────────────────────────


async def _tool_ltm_query(args: dict) -> str:
    from sentence_transformers import SentenceTransformer

    query = str(args["query"])
    top_k = max(1, min(int(args.get("top_k", 5)), 20))

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding = json.dumps(embedder.encode(query, normalize_embeddings=True).tolist())

    import asyncpg as apg

    conn = await apg.connect(_pg_dsn())
    try:
        rows = await conn.fetch(
            "SELECT community_id, summary, entity_count, hub_entities FROM ltm_topics "
            f"ORDER BY embedding <=> $1::vector LIMIT {top_k}",  # nosec B608
            embedding,
        )
    finally:
        await conn.close()

    results = []
    for r in rows:
        results.append(
            {
                "community_id": r["community_id"],
                "summary": r["summary"],
                "entity_count": r["entity_count"],
                "hub_entities": r.get("hub_entities"),
            }
        )
    return json.dumps(results, ensure_ascii=False, indent=2)


async def _tool_ltm_entity(args: dict) -> str:
    name = str(args["name"])
    d = _neo4j_driver()
    try:
        with d.session() as s:
            row = s.run(
                "MATCH (e:Entity {name: $name}) "
                "OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:Segment) "
                "RETURN properties(e) AS props, count(DISTINCT seg) AS segment_count",
                name=name,
            ).single()
    finally:
        d.close()

    if row is None:
        return json.dumps({"error": f"Entity '{name}' not found"})

    props = dict(row["props"])
    props["segment_count"] = row["segment_count"]
    return json.dumps(props, ensure_ascii=False, indent=2, default=str)


async def _tool_ltm_topics(args: dict) -> str:
    min_entities = int(args.get("min_entities", 3))
    d = _neo4j_driver()
    try:
        with d.session() as s:
            rows = s.run(
                "MATCH (t:Topic) "
                "WHERE t.entity_count >= $min_entities "
                "RETURN t.community_id AS community_id, "
                "       t.summary AS summary, "
                "       t.entity_count AS entity_count "
                "ORDER BY t.entity_count DESC",
                min_entities=min_entities,
            ).data()
    finally:
        d.close()

    return json.dumps(rows, ensure_ascii=False, indent=2)


async def _tool_recall_by_document(args: dict) -> str:
    from graphknows.recall import recall_by_document as _rbd

    query = str(args["query"])
    top_k = max(1, min(int(args.get("top_k", 5)), 20))
    d = _neo4j_driver()
    try:
        results = _rbd(d, query, top_k)
    finally:
        d.close()
    return json.dumps(
        [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
    )


async def _tool_recall_by_dialog(args: dict) -> str:
    from graphknows.recall import recall_by_dialog as _rbdia

    query = str(args["query"])
    top_k = max(1, min(int(args.get("top_k", 5)), 20))
    session_filter = args.get("session_filter") or None
    d = _neo4j_driver()
    try:
        results = _rbdia(d, query, top_k, session_filter)
    finally:
        d.close()
    return json.dumps(
        [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
    )


async def _tool_recall_by_topic(args: dict) -> str:
    from graphknows.recall import recall_by_topic as _rbt

    query = str(args["query"])
    top_k = max(1, min(int(args.get("top_k", 5)), 20))
    d = _neo4j_driver()
    try:
        results = _rbt(d, query, top_k)
    finally:
        d.close()
    return json.dumps(
        [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
    )


# ── Entry point ───────────────────────────────────────────────────────────


async def mcp_main() -> None:
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(mcp_main())
