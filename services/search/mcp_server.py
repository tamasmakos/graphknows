"""
MCP Server — GraphKnows EU Parliamentary Search.

Exposes EU document search as a stdio MCP server compatible with
Claude's runSubagent, Cursor, and any MCP client.

Tools
-----
  search_eu_documents  Search across EU parliamentary & legislative sources
  fetch_document       Retrieve full document text by ID
  list_sources         List all available EU data sources with descriptions
  query_sparql         Run a SPARQL query against allowlisted EU endpoints

Resources
---------
  source://{source_id}   Metadata card for a specific source connector

Prompts
-------
  eu_research_assistant   Pre-filled system+user prompt for EU research sessions

Run via
-------
    SEARCH_MODE=mcp python -m services.search.main
    # or directly:
    python -m services.search.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .connector_registry import ConnectorRegistry, get_registry
from .archive_engine import ArchiveEngine, ArchiveQuery
from .models import SearchQuery

log = logging.getLogger(__name__)

# ── SPARQL endpoint allowlist (SSRF prevention — never expose raw URLs to callers) ──
_SPARQL_ENDPOINTS: dict[str, str] = {
    "cellar": "https://publications.europa.eu/webapi/rdf/sparql",
    "data_europa": "https://data.europa.eu/sparql",
}

# ── MCP Server ────────────────────────────────────────────────────────────────

server = Server("graphknows-eu-search")
_engine = ArchiveEngine()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_eu_documents",
            description=(
                "Search across European parliamentary and public data sources. "
                "Returns ranked excerpts from EP debates, EUR-Lex legislation, "
                "Council conclusions, ECB publications, EPRS briefings, and more. "
                "Specify sources to target specific EU institutions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of source IDs to search. "
                            "Omit to search all enabled sources. "
                            "Use list_sources to discover available IDs."
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": "ISO date lower bound e.g. 2023-01-01",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "ISO date upper bound e.g. 2024-12-31",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 8,
                        "description": "Max results to return (1–20)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="fetch_document",
            description=(
                "Retrieve the full text of a specific EU document by its ID. "
                "Use after search_eu_documents to get complete document content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source connector ID (e.g. 'eurlex', 'ep_debates')",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Document ID as returned by search_eu_documents",
                    },
                },
                "required": ["source_id", "document_id"],
            },
        ),
        types.Tool(
            name="list_sources",
            description=(
                "List all available EU data source connectors with their descriptions, "
                "coverage periods, jurisdictions, and query strengths. "
                "Use this to choose which sources to target for a specific research task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "jurisdiction": {
                        "type": "string",
                        "description": "Filter by jurisdiction code: EU, UK, GLOBAL, MEMBER_STATES",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Filter by document type e.g. legislation, press_release",
                    },
                },
            },
        ),
        types.Tool(
            name="query_sparql",
            description=(
                "Run a read-only SPARQL SELECT query against allowlisted EU linked-data "
                "endpoints. Supports the EU CELLAR CDM ontology. "
                "endpoint_id must be one of: " + ", ".join(_SPARQL_ENDPOINTS.keys())
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_id": {
                        "type": "string",
                        "enum": list(_SPARQL_ENDPOINTS.keys()),
                        "description": "Endpoint alias (cellar | data_europa)",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "SPARQL SELECT query string",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max rows to return (1–100)",
                    },
                },
                "required": ["endpoint_id", "sparql"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = await _dispatch(name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as exc:
        log.exception("[mcp] tool %s failed", name)
        return [types.TextContent(type="text", text=f"Error: {exc}")]


async def _dispatch(name: str, args: dict) -> str:
    if name == "search_eu_documents":
        return await _tool_search(args)
    if name == "fetch_document":
        return await _tool_fetch(args)
    if name == "list_sources":
        return _tool_list_sources(args)
    if name == "query_sparql":
        return await _tool_query_sparql(args)
    raise ValueError(f"Unknown tool: {name}")


# ── Tool implementations ──────────────────────────────────────────────────────


async def _tool_search(args: dict) -> str:
    query = str(args["query"])
    registry = get_registry()
    sources: list[str] = args.get("sources") or registry.list_ids()
    limit = max(1, min(int(args.get("limit", 8)), 20))

    aq = ArchiveQuery(
        query=query,
        archives=sources,
        date_from=args.get("date_from"),
        date_to=args.get("date_to"),
        limit_per_archive=max(2, limit // max(len(sources), 1)),
        top_k=limit,
    )
    results = await _engine.search(aq)
    return _engine.format_for_agent(results)


async def _tool_fetch(args: dict) -> str:
    source_id = str(args["source_id"])
    document_id = str(args["document_id"])
    registry = get_registry()
    connector = registry.get(source_id)
    if connector is None:
        raise ValueError(f"Unknown source: {source_id}")
    doc = await connector.fetch(document_id)
    return (
        f"# {doc.title}\n\n"
        f"Source: {source_id}\n"
        f"ID: {document_id}\n"
        f"URL: {doc.metadata.get('url', 'N/A')}\n"
        f"Date: {doc.metadata.get('date', 'N/A')}\n\n"
        f"{doc.text}"
    )


def _tool_list_sources(args: dict) -> str:
    registry = get_registry()
    jurisdiction = args.get("jurisdiction")
    doc_type = args.get("doc_type")
    entries = registry.list_entries(jurisdiction=jurisdiction, doc_type=doc_type)
    if not entries:
        return "No sources match the specified filters."
    lines: list[str] = [f"Found {len(entries)} sources:\n"]
    for e in entries:
        strengths = ", ".join(e.query_strengths[:3])
        lines.append(
            f"**{e.id}** — {e.name}\n"
            f"  {e.description.strip()[:200]}\n"
            f"  Coverage: {e.temporal_coverage} | "
            f"Lang: {e.language} | "
            f"Jurisdictions: {', '.join(e.jurisdictions)}\n"
            f"  Best for: {strengths}"
        )
    return "\n\n".join(lines)


async def _tool_query_sparql(args: dict) -> str:
    endpoint_id = str(args["endpoint_id"])
    sparql_query = str(args["sparql"])
    limit = max(1, min(int(args.get("limit", 20)), 100))

    # Validate endpoint via allowlist (SSRF prevention)
    endpoint_url = _SPARQL_ENDPOINTS.get(endpoint_id)
    if endpoint_url is None:
        raise ValueError(
            f"Unknown endpoint '{endpoint_id}'. Allowed: {list(_SPARQL_ENDPOINTS.keys())}"
        )

    # Only allow SELECT queries — reject mutations (SPARQL injection prevention)
    stripped = sparql_query.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("PREFIX"):
        raise ValueError("Only SPARQL SELECT queries are allowed.")
    if any(kw in stripped for kw in ("INSERT", "DELETE", "UPDATE", "DROP", "CLEAR", "LOAD")):
        raise ValueError("Write operations are not permitted.")

    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            endpoint_url,
            data={"query": sparql_query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        data = resp.json()

    bindings = data.get("results", {}).get("bindings", [])[:limit]
    if not bindings:
        return "Query returned no results."

    # Format as markdown table
    if bindings:
        vars_ = list(bindings[0].keys())
        header = " | ".join(vars_)
        sep = " | ".join(["---"] * len(vars_))
        rows = [" | ".join(b.get(v, {}).get("value", "") for v in vars_) for b in bindings]
        return "\n".join([header, sep] + rows)
    return "No results."


# ── Resources ─────────────────────────────────────────────────────────────────


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    registry = get_registry()
    return [
        types.Resource(
            uri=f"source://{e.id}",
            name=e.name,
            description=e.description.strip()[:200],
            mimeType="text/plain",
        )
        for e in registry.list_entries()
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if not uri.startswith("source://"):
        raise ValueError(f"Unknown resource URI: {uri}")
    source_id = uri[len("source://") :]
    registry = get_registry()
    entries = {e.id: e for e in registry.list_entries()}
    entry = entries.get(source_id)
    if entry is None:
        raise ValueError(f"Source not found: {source_id}")
    return (
        f"# {entry.name}\n\n"
        f"{entry.description.strip()}\n\n"
        f"- **ID**: {entry.id}\n"
        f"- **Jurisdictions**: {', '.join(entry.jurisdictions)}\n"
        f"- **Document types**: {', '.join(entry.doc_types)}\n"
        f"- **Temporal coverage**: {entry.temporal_coverage}\n"
        f"- **Language**: {entry.language}\n"
        f"- **Rate limit**: {entry.rate_limit_rps} req/s\n"
        f"- **Requires auth**: {entry.requires_auth}\n\n"
        f"**Best for**: {', '.join(entry.query_strengths)}"
    )


# ── Prompts ───────────────────────────────────────────────────────────────────


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="eu_research_assistant",
            description=(
                "System prompt that configures an agent for EU parliamentary research. "
                "Instructs the model to use the EU search tools systematically."
            ),
            arguments=[
                types.PromptArgument(
                    name="topic",
                    description="The EU policy topic to research",
                    required=True,
                )
            ],
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name != "eu_research_assistant":
        raise ValueError(f"Unknown prompt: {name}")
    topic = (arguments or {}).get("topic", "EU policy")
    return types.GetPromptResult(
        description="EU Parliamentary Research Assistant",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=(
                        f"You are an EU parliamentary research assistant. "
                        f"Research the following topic: **{topic}**\n\n"
                        f"Workflow:\n"
                        f"1. Call list_sources to identify the most relevant EU data sources\n"
                        f"2. Call search_eu_documents with targeted sources and date filters\n"
                        f"3. Call fetch_document for the most relevant results to get full text\n"
                        f"4. Optionally call query_sparql for structured CDM data from CELLAR\n"
                        f"5. Synthesise findings into a concise, cited research brief\n\n"
                        f"Always cite source IDs, document IDs, and URLs in your output."
                    ),
                ),
            )
        ],
    )


# ── Entry point ───────────────────────────────────────────────────────────────


async def mcp_main() -> None:
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(mcp_main())
