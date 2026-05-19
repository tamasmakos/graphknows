"""graphknows FastMCP server — 13 agentic tools across retrieval, graph, and memory.

Tools
-----
retrieve             — hybrid/semantic/graph/summaries/lucky search → XML
entity_search        — ANN entity discovery with optional label filter → JSON
entity_neighbors     — graph traversal from a named entity → JSON
get_chunks           — batch chunk fetch with provenance → JSON
session_memory       — store text into session-scoped vector store → JSON
ingest               — trigger graphgen corpus pipeline → run_id
poll_job             — check ingest job status → JSON
get_session_graph    — Memgraph session entity graph → JSON
get_corpus_stats     — Neo4j + pgvector counts → JSON
community_summary    — Leiden topic summary for a named entity → text
list_segments        — list SEGMENT provenance nodes → text
segment_info         — provenance detail for one SEGMENT node → text
cypher_query         — parameterized read-only Cypher → JSON
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("graphknows.mcp")

# ---------------------------------------------------------------------------
# Shared job store — imported by api.py to share the same reference
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Singletons injected from FastAPI lifespan via set_dependencies()
# ---------------------------------------------------------------------------

_retriever: Any = None  # HybridRetriever
_pg: Any = None         # PgVectorStore
_neo: Any = None        # Neo4jStore
_memgraph: Any = None   # MemgraphStore
_cfg: dict = {}

# ---------------------------------------------------------------------------
# Per-client session scoping
# Mapping: client_id -> session_id (volatile, single-process only)
# ---------------------------------------------------------------------------

_client_sessions: dict[str, str] = {}


def set_dependencies(
    retriever: Any,
    pg: Any,
    neo: Any,
    memgraph: Any,
    cfg: dict,
) -> None:
    """Register shared store instances for all MCP tools.

    Called from FastAPI lifespan after all stores are initialised.

    Note:
        The client_id->session_id scoping map is in-memory and not shared
        across worker processes.  Use explicit session_id for multi-process
        deployments.
    """
    global _retriever, _pg, _neo, _memgraph, _cfg
    _retriever = retriever
    _pg = pg
    _neo = neo
    _memgraph = memgraph
    _cfg = cfg


def _require_retriever() -> Any:
    if _retriever is None:
        raise RuntimeError("graphknows retriever not yet initialised")
    return _retriever


def _require_pg() -> Any:
    if _pg is None:
        raise RuntimeError("pgvector store not yet initialised")
    return _pg


def _require_neo() -> Any:
    if _neo is None:
        raise RuntimeError("Neo4j store not yet initialised")
    return _neo


def _require_memgraph() -> Any:
    if _memgraph is None:
        raise RuntimeError("Memgraph store not yet initialised")
    return _memgraph


def _resolve_session(session_id: str | None, client_id: str | None) -> str:
    """Resolve the effective session_id for a tool call.

    Priority:
    1. Explicit session_id always wins.
    2. client_id present — auto-create/reuse a client-scoped session.
    3. Neither provided -> "default" (shared, no isolation).
    """
    if session_id:
        return session_id
    if client_id:
        if client_id not in _client_sessions:
            _client_sessions[client_id] = f"client:{client_id}:{uuid.uuid4().hex[:8]}"
            logger.info(
                "Auto-created session %s for client %r",
                _client_sessions[client_id],
                client_id,
            )
        return _client_sessions[client_id]
    return "default"


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "graphknows",
    instructions=(
        "graphknows agentic knowledge graph tools.\n\n"
        "Quick-start workflow:\n"
        "1. Call `retrieve` with your research question (default HYBRID mode).\n"
        "2. Use `entity_search` to discover relevant named entities.\n"
        "3. Use `entity_neighbors` to traverse the graph around an entity.\n"
        "4. Use `community_summary` to get an LLM-generated topic overview.\n"
        "5. Use `ingest` + `poll_job` to add new documents to the corpus.\n"
        "6. Use `session_memory` to store working notes in your session.\n\n"
        "All tools accept an optional `client_id` (e.g. 'cursor', 'claude') to\n"
        "automatically scope sessions — you do not need to manage session_id manually."
    ),
)


# ---------------------------------------------------------------------------
# Tool: retrieve (multi-mode)
# ---------------------------------------------------------------------------


@mcp.tool()
async def retrieve(
    query: str,
    session_id: str | None = None,
    search_type: str = "HYBRID",
    k_entity: int = 10,
    k_semantic: int = 20,
    k_final: int = 10,
    client_id: str | None = None,
) -> str:
    """Search the knowledge graph using the specified strategy.

    Returns an XML document conforming to the <retrieval_context> schema.
    For Cypher queries use the dedicated `cypher_query` tool.

    Search modes:
    - HYBRID (default): RRF fusion of entity-graph + semantic vector search.
    - SEMANTIC: Pure pgvector ANN on session-scoped chunks (fastest).
    - GRAPH_TRAVERSAL: Entity ANN -> multi-hop graph expansion -> linked chunks.
    - SUMMARIES: Entity ANN -> Leiden community topic summaries.
    - FEELING_LUCKY: Auto-selects the best mode based on query heuristics.

    Args:
        query: Natural-language research question.
        session_id: Session namespace. Omit to auto-scope by client_id.
        search_type: HYBRID | SEMANTIC | GRAPH_TRAVERSAL | SUMMARIES | FEELING_LUCKY.
        k_entity: Entities to retrieve (HYBRID / GRAPH_TRAVERSAL / SUMMARIES).
        k_semantic: Chunks retrieved by direct ANN (HYBRID / SEMANTIC).
        k_final: Maximum results in the fused output.
        client_id: Optional agent identity for auto-session scoping.
    """
    ret = _require_retriever()
    sid = _resolve_session(session_id, client_id)
    mode = search_type.upper()

    if mode == "SEMANTIC":
        ctx = await ret.retrieve_semantic(query, sid, k=k_semantic)
    elif mode == "GRAPH_TRAVERSAL":
        ctx = await ret.retrieve_graph_traversal(
            query, sid, k_entity=k_entity, depth=2, k_final=k_final
        )
    elif mode == "SUMMARIES":
        ctx = await ret.retrieve_summaries(query, sid, k_entity=k_entity)
    elif mode == "FEELING_LUCKY":
        ctx = await ret.retrieve_feeling_lucky(
            query, sid, k_entity=k_entity, k_semantic=k_semantic, k_final=k_final
        )
    elif mode == "HYBRID":
        ctx = await ret.retrieve(
            query, sid, k_entity=k_entity, k_semantic=k_semantic, k_final=k_final
        )
    else:
        raise ValueError(
            f"Unknown search_type {search_type!r}. "
            "Valid: HYBRID, SEMANTIC, GRAPH_TRAVERSAL, SUMMARIES, FEELING_LUCKY. "
            "For Cypher use the `cypher_query` tool."
        )
    return ctx.to_xml()


# ---------------------------------------------------------------------------
# Tool: entity_search
# ---------------------------------------------------------------------------


@mcp.tool()
async def entity_search(
    query: str,
    k: int = 10,
    entity_types: list[str] | None = None,
    client_id: str | None = None,
) -> str:
    """Discover named entities in the corpus by semantic similarity.

    Embeds the query and returns the closest entities from the corpus entity
    table.  Optionally filter by entity type label.

    Prerequisites: corpus must have been ingested (run `ingest` first).

    Args:
        query: Natural-language description or entity name to search for.
        k: Maximum entities to return (default 10).
        entity_types: Optional label filter, e.g. ["ORG", "PERSON", "LOC"].
        client_id: Agent identity (informational).

    Returns:
        JSON array of {name, label, score} objects sorted by similarity.
    """
    from graphknows.retrieval.stores.embedder import embed

    pg = _require_pg()
    emb_array = await asyncio.to_thread(
        embed, [query], _cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    )
    emb = emb_array[0].tolist()
    results = await pg.search_entities(emb, k=k, entity_types=entity_types or None)
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: entity_neighbors
# ---------------------------------------------------------------------------


@mcp.tool()
async def entity_neighbors(
    entity_name: str,
    depth: int = 1,
    relationship_types: list[str] | None = None,
    client_id: str | None = None,
) -> str:
    """Traverse the knowledge graph from a named entity.

    Returns neighboring Entity and Topic nodes up to depth hops away.
    Depth is capped at 3 to prevent runaway graph scans.

    Args:
        entity_name: Starting entity name (case-insensitive).
        depth: Number of hops to traverse (1-3, default 1).
        relationship_types: Optional allowlist of relationship labels.
            Only alphanumeric + underscore names accepted.
        client_id: Agent identity (informational).

    Returns:
        JSON dict with entity, depth, neighbor_count, and neighbors list.
    """
    neo = _require_neo()
    result = await neo.get_entity_neighbors(
        entity_name,
        depth=depth,
        relationship_types=relationship_types,
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: get_chunks
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_chunks(
    chunk_ids: list[str],
    client_id: str | None = None,
) -> str:
    """Fetch full chunk text and provenance for a list of chunk IDs.

    Use this after `retrieve` to inspect raw text and linked entities for
    specific chunks from the fused_results.

    Args:
        chunk_ids: List of chunk ID strings (from retrieve results).
        client_id: Agent identity (informational).

    Returns:
        JSON dict mapping chunk_id -> {doc_id, text, metadata, entities}.
    """
    pg = _require_pg()
    neo = _require_neo()

    text_map = await pg.get_chunks_by_ids(chunk_ids)
    entity_tasks = [neo.get_chunk_entities(cid) for cid in chunk_ids]
    entity_results = await asyncio.gather(*entity_tasks, return_exceptions=True)

    enriched: dict[str, Any] = {}
    for cid, ent_result in zip(chunk_ids, entity_results):
        base = text_map.get(cid, {})
        enriched[cid] = {
            "doc_id": base.get("doc_id", ""),
            "text": base.get("text", ""),
            "metadata": base.get("metadata", {}),
            "entities": ent_result if not isinstance(ent_result, Exception) else [],
        }
    return json.dumps(enriched, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: session_memory
# ---------------------------------------------------------------------------


@mcp.tool()
async def session_memory(
    content: str,
    title: str = "",
    session_id: str | None = None,
    client_id: str | None = None,
) -> str:
    """Store a text note in the session-scoped vector store.

    The content is embedded and upserted as a chunk under the resolved session,
    making it retrievable via `retrieve` with SEMANTIC or HYBRID mode.

    Args:
        content: Text content to store.
        title: Optional title prefix (prepended before embedding).
        session_id: Session namespace. Omit to auto-scope by client_id.
        client_id: Agent identity for session auto-scoping.

    Returns:
        JSON dict confirming chunk_id stored and the resolved session_id.
    """
    from graphknows.retrieval.stores.embedder import embed

    pg = _require_pg()
    sid = _resolve_session(session_id, client_id)
    full_text = f"{title}\n\n{content}" if title else content
    chunk_id = f"memo:{uuid.uuid4().hex}"

    emb_array = await asyncio.to_thread(
        embed, [full_text], _cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    )
    emb = emb_array[0].tolist()

    await pg.upsert_chunks(
        session_id=sid,
        chunks=[
            {
                "id": chunk_id,
                "doc_id": f"memo:{sid}",
                "text": full_text,
                "embedding": emb,
                "metadata": {"title": title, "source": "session_memory"},
            }
        ],
    )
    return json.dumps({"chunk_id": chunk_id, "session_id": sid, "stored": True})


# ---------------------------------------------------------------------------
# Tool: ingest
# ---------------------------------------------------------------------------


async def _run_mcp_ingest_task(
    run_id: str, text: str, title: str, config_override: dict
) -> None:
    """Background ingestion task — updates shared _jobs dict."""
    _jobs[run_id]["status"] = "running"
    _jobs[run_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        from graphknows.graphgen import run_ingestion

        stats = await run_ingestion(
            documents=[{"text": text, "title": title}],
            config_override=config_override or None,
        )
        _jobs[run_id]["status"] = "completed"
        _jobs[run_id]["stats"] = stats
        _jobs[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("MCP ingest job %s completed", run_id)
    except Exception as exc:
        logger.exception("MCP ingest job %s failed: %s", run_id, exc)
        _jobs[run_id]["status"] = "failed"
        _jobs[run_id]["error"] = str(exc)
        _jobs[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@mcp.tool()
async def ingest(
    text: str,
    title: str = "",
    config_override: dict | None = None,
    client_id: str | None = None,
) -> str:
    """Add a document to the persistent corpus knowledge graph (non-blocking).

    Enqueues a graphgen KnowledgePipeline run in the background. The pipeline
    performs NER, entity resolution, community detection, and Neo4j upload.
    Use `poll_job` with the returned run_id to check completion.

    Args:
        text: Document body text.
        title: Optional document title.
        config_override: Optional dict overriding pipeline settings.
        client_id: Agent identity (informational).

    Returns:
        JSON dict with run_id and status="queued".
    """
    run_id = str(uuid.uuid4())
    queued_at = datetime.now(timezone.utc).isoformat()
    _jobs[run_id] = {"status": "queued", "queued_at": queued_at}
    asyncio.create_task(
        _run_mcp_ingest_task(run_id, text, title, config_override or {})
    )
    return json.dumps({"run_id": run_id, "status": "queued", "queued_at": queued_at})


# ---------------------------------------------------------------------------
# Tool: poll_job
# ---------------------------------------------------------------------------


@mcp.tool()
async def poll_job(run_id: str) -> str:
    """Check the status of an ingest job started by `ingest`.

    Args:
        run_id: The run_id returned by `ingest`.

    Returns:
        JSON dict with status (queued | running | completed | failed),
        stats on completion, or error on failure.
    """
    job = _jobs.get(run_id)
    if job is None:
        return json.dumps({"run_id": run_id, "status": "not_found"})
    return json.dumps({"run_id": run_id, **job})


# ---------------------------------------------------------------------------
# Tool: get_session_graph
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_session_graph(
    session_id: str | None = None,
    client_id: str | None = None,
) -> str:
    """Return the entity graph for the current session from Memgraph.

    Shows all entities and RELATES edges stored during the active session.
    Useful for auditing session context before merging to the corpus.

    Args:
        session_id: Session namespace. Omit to auto-scope by client_id.
        client_id: Agent identity for session auto-scoping.

    Returns:
        JSON dict with session_id, node_count, edge_count, nodes, and edges.
    """
    mg = _require_memgraph()
    sid = _resolve_session(session_id, client_id)
    result = await mg.get_session_graph(sid)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: get_corpus_stats
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_corpus_stats() -> str:
    """Return combined node and row counts from Neo4j and pgvector.

    Returns:
        JSON dict with neo4j {chunks, entities, topics, session_entities}
        and pgvector {chunks, entities} counts.
    """
    neo = _require_neo()
    pg = _require_pg()

    neo_counts = await neo.count_nodes()
    pool = pg._pool  # noqa: SLF001
    pg_counts: dict[str, int] = {"chunks": 0, "entities": 0}
    if pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT (SELECT count(*) FROM chunks) AS chunks, "
                "(SELECT count(*) FROM entities) AS entities"
            )
            if rows:
                pg_counts = {
                    "chunks": int(rows[0]["chunks"]),
                    "entities": int(rows[0]["entities"]),
                }

    return json.dumps({"neo4j": neo_counts, "pgvector": pg_counts})


# ---------------------------------------------------------------------------
# Tool: community_summary
# ---------------------------------------------------------------------------


@mcp.tool()
async def community_summary(
    entity_name: str,
    client_id: str | None = None,
) -> str:
    """Return the Leiden community topic summary for a named entity.

    Looks up the Topic node linked to entity_name via IN_COMMUNITY edges
    and returns the LLM-generated topic summary.

    Args:
        entity_name: Entity name to look up (case-insensitive).
        client_id: Agent identity (informational).

    Returns:
        Topic label + summary string, or a message if no community found.
    """
    neo = _require_neo()
    topic = await neo.get_entity_community(entity_name)
    if topic is None:
        return f"No community found for entity: {entity_name}"
    return f"{topic['label']}\n\n{topic['summary']}"


# ---------------------------------------------------------------------------
# Tool: list_segments
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_segments(limit: int = 20) -> str:
    """List temporal SEGMENT nodes (pipeline-run provenance containers).

    Returns a text summary of the most recent limit SEGMENT nodes ordered
    by ingestion timestamp descending.

    Args:
        limit: Maximum number of segments to return (default 20).

    Returns:
        Human-readable text listing segment_id, ingested_at, and taxonomy.
    """
    neo = _require_neo()
    segments = await neo.list_segments(limit=limit)
    if not segments:
        return "No SEGMENT nodes found in the corpus."
    lines = ["SEGMENT nodes (newest first):", ""]
    for seg in segments:
        lines.append(
            f"  {seg.get('segment_id', 'unknown')}  |  {seg.get('ingested_at', 'N/A')}"
            f"  |  taxonomy={seg.get('taxonomy_name', 'N/A')}"
            f"  |  docs={seg.get('source_count', '?')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: segment_info
# ---------------------------------------------------------------------------


@mcp.tool()
async def segment_info(segment_id: str) -> str:
    """Return provenance and statistics for a single SEGMENT pipeline-run node.

    Args:
        segment_id: The segment_id attribute of the SEGMENT node.

    Returns:
        Human-readable provenance string, or a not-found message.
    """
    neo = _require_neo()
    info = await neo.get_segment_info(segment_id)
    if info is None:
        return f"Segment {segment_id!r} not found."
    lines = [f"SEGMENT: {segment_id}", ""]
    for key, val in info.items():
        if val is not None:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: cypher_query
# ---------------------------------------------------------------------------

_CYPHER_ENABLED: bool | None = None


def _cypher_enabled() -> bool:
    global _CYPHER_ENABLED
    if _CYPHER_ENABLED is None:
        import os
        _CYPHER_ENABLED = os.getenv("ENABLE_CYPHER_TOOL", "false").lower() in (
            "1", "true", "yes"
        )
    return _CYPHER_ENABLED


@mcp.tool()
async def cypher_query(
    cypher_template: str,
    params: dict | None = None,
) -> str:
    """Execute a parameterized read-only Cypher query against the Neo4j corpus.

    Only available when ENABLE_CYPHER_TOOL=true is set in the environment.
    Write operations (CREATE, MERGE, SET, DELETE, etc.) are always rejected.

    User values must be passed via params as $param references — never
    interpolated into the template string. Templates containing {var} f-string
    markers are rejected outright.

    Example:
        cypher_template = "MATCH (e:Entity {name: $name}) RETURN e.label AS label"
        params = {"name": "EUROPOL"}

    Args:
        cypher_template: Cypher query with $param placeholders.
        params: Dict of parameter values bound to $param references.

    Returns:
        JSON array of row dicts, or error JSON if the tool is disabled.
    """
    if not _cypher_enabled():
        return json.dumps(
            {
                "error": "cypher_query is disabled. "
                "Set ENABLE_CYPHER_TOOL=true to enable read-only parameterized Cypher."
            }
        )
    ret = _require_retriever()
    rows = await ret.execute_cypher(cypher_template, params)
    return json.dumps(rows, ensure_ascii=False, default=str)
