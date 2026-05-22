"""graphknows — unified agentic memory service.

Endpoints
---------
GET  /health                         — liveness probe
POST /ingest                         — enqueue graphgen KnowledgePipeline (background task)
GET  /jobs/{run_id}                  — poll ingest job status
POST /retrieve                       — hybrid entity-graph + semantic retrieval → XML
POST /session                        — create or re-seed a session namespace
DELETE /session/{session_id}         — merge Memgraph session → Neo4j corpus and purge
GET  /session/{session_id}/preview-merge — preview a session merge without writing
GET  /stats                          — combined Neo4j + pgvector row counts
POST /admin/patch                    — run ontology patch phases on Neo4j corpus
GET  /admin/patch/preview            — dry-run patch, returns preview
GET  /segments                       — list SEGMENT provenance nodes
GET  /segments/{segment_id}          — provenance + stats for one SEGMENT node
MCP  /mcp/                           — FastMCP tool endpoint (retrieve, community_summary)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger

from graphknows.config import GraphKnowsSettings
from graphknows.mcp.server import _jobs, mcp, set_dependencies
from graphknows.retrieval.retriever import HybridRetriever
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.neo4j import Neo4jStore
from graphknows.retrieval.stores.pgvector import PgVectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("graphknows.api")

# Module-level constants for repeated HTTPException detail strings
_ERR_STORES_NOT_INIT = "Stores not initialised"
_ERR_NEO4J_NOT_INIT = "Neo4j not initialised"

# Reusable responses dicts for FastAPI OpenAPI schema documentation
_DESC_503: str = "Service unavailable"
_R503: dict[int | str, dict[str, Any]] = {503: {"description": _DESC_503}}
_R503_404: dict[int | str, dict[str, Any]] = {
    503: {"description": _DESC_503},
    404: {"description": "Not found"},
}
_R503_500: dict[int | str, dict[str, Any]] = {
    503: {"description": _DESC_503},
    500: {"description": "Internal error"},
}
_R503_422: dict[int | str, dict[str, Any]] = {
    503: {"description": _DESC_503},
    422: {"description": "Invalid input"},
}
_R404: dict[int | str, dict[str, Any]] = {404: {"description": "Not found"}}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# GraphKnowsSettings is now in graphknows/config.py — imported above.

# In-memory job store now lives in mcp/server.py (_jobs) and is imported above.
# This keeps MCP ingest tool and REST ingest endpoint on the same shared store.

# ---------------------------------------------------------------------------
# Session TTL tracking (E5)
# ---------------------------------------------------------------------------

# Maps session_id → last activity timestamp
_session_last_active: dict[str, datetime] = {}

_ttl_task: asyncio.Task[None] | None = None


async def _session_ttl_loop() -> None:
    """Background loop: auto-merge sessions idle longer than SESSION_TTL_MINUTES."""
    global _session_last_active
    while True:
        await asyncio.sleep(60)  # check every minute
        if _settings is None or _neo is None or _memgraph is None or _pg is None:
            continue
        ttl = timedelta(minutes=_settings.session_ttl_minutes)
        now = datetime.now(UTC)
        expired = [sid for sid, ts in _session_last_active.items() if now - ts > ttl]
        for session_id in expired:
            logger.info("Auto-merging idle session %s (TTL exceeded)", session_id)
            try:
                nodes = await _memgraph.get_session_nodes(session_id)
                edges = await _memgraph.get_session_edges(session_id)
                await _neo.merge_session_entities_to_corpus(session_id, nodes, edges)
                await asyncio.gather(
                    _memgraph.delete_session(session_id),
                    _pg.delete_session(session_id),
                )
                _session_last_active.pop(session_id, None)
                logger.info("Auto-merge of session %s complete", session_id)
            except Exception as exc:
                logger.exception("Auto-merge failed for session %s: %s", session_id, exc)


# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------

_pg: PgVectorStore | None = None
_neo: Neo4jStore | None = None
_memgraph: MemgraphStore | None = None
_retriever: HybridRetriever | None = None
_settings: GraphKnowsSettings | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pg, _neo, _memgraph, _retriever, _settings, _ttl_task

    _settings = GraphKnowsSettings()
    logger.info("graphknows starting", extra={"env": _settings.graphknows_env})

    _neo = Neo4jStore(
        uri=_settings.neo4j_uri,
        user=_settings.neo4j_user,
        password=_settings.neo4j_password,
        database=_settings.neo4j_database,
    )
    _pg = PgVectorStore(dsn=_settings.pgvector_dsn)
    _memgraph = MemgraphStore(uri=_settings.memgraph_uri)

    try:
        await _pg.connect()
        await asyncio.gather(
            _pg.ensure_schema(),
            _neo.ensure_indexes(),
        )
        _retriever = HybridRetriever(
            pg=_pg,
            neo=_neo,
            embed_model=_settings.embed_model,
        )
        set_dependencies(_retriever, _pg, _neo, _memgraph, {"embed_model": _settings.embed_model})
        logger.info("graphknows ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("graphknows started in degraded mode: %s", exc)

    # Start session TTL loop (E5)
    _ttl_task = asyncio.create_task(_session_ttl_loop())

    yield

    logger.info("graphknows shutting down")
    if _ttl_task is not None:
        _ttl_task.cancel()
    try:
        await _pg.close()
    except Exception:
        pass
    try:
        _neo.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="graphknows",
    description="Unified agentic memory: graph ingestion + hybrid retrieval",
    version="1.0.0",
    lifespan=_lifespan,
)

# Mount FastMCP at /mcp/
app.mount("/mcp", mcp.streamable_http_app())


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DocumentIn(BaseModel):
    """A document to ingest."""

    id: str | None = Field(
        default=None, description="Optional document ID (auto-generated if omitted)"
    )
    text: str = Field(..., description="Document body text")
    title: str | None = Field(default=None, description="Optional document title")
    source: str | None = Field(default=None, description="Optional source URL or label")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    documents: list[DocumentIn] = Field(..., min_length=1)
    config_override: dict[str, Any] | None = Field(default=None)
    output_dir: str | None = Field(default=None)


class IngestResponse(BaseModel):
    """Response returned when an ingestion job is queued."""

    run_id: str
    status: str
    queued_at: str


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve."""

    query: str = Field(..., min_length=1)
    session_id: str
    k_entity: int = Field(default=10, ge=1, le=100)
    k_semantic: int = Field(default=20, ge=1, le=200)
    k_final: int = Field(default=10, ge=1, le=100)


class SessionCreateRequest(BaseModel):
    """Request body for POST /session."""

    session_id: str | None = Field(default=None, description="Optional — auto-generated if omitted")
    seed_entities: list[str] = Field(
        default_factory=list,
        description="Entity names to pre-load from the corpus into the session graph",
    )


class SessionCreateResponse(BaseModel):
    """Response returned when a session is created."""

    session_id: str
    seeded: int


class SessionDeleteResponse(BaseModel):
    """Response returned when a session is deleted and its graph merged."""

    session_id: str
    merged_nodes: int
    merged_edges: int
    purged_chunks: int


class PatchRequest(BaseModel):
    """Request body for POST /admin/patch."""

    phases: list[str] | None = Field(
        default=None,
        description="Phases to run; omit to run all default phases",
    )
    dry_run: bool = Field(default=False, description="Preview changes without writing")
    batch_size: int = Field(default=500, ge=1, le=10_000)
    taxonomy_file: str | None = Field(
        default=None,
        description="Path to taxonomy YAML; uses bundled eu-politics.yaml if omitted",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "graphknows"}


# ── Ingest ─────────────────────────────────────────────────────────────────


async def _run_ingest_task(run_id: str, req: IngestRequest) -> None:
    """Background task: run graphgen pipeline and update job store."""
    _jobs[run_id]["status"] = "running"
    _jobs[run_id]["started_at"] = datetime.now(UTC).isoformat()
    try:
        from graphknows.graphgen import run_ingestion

        docs = [d.model_dump() for d in req.documents]
        stats = await run_ingestion(
            documents=docs,
            config_override=req.config_override,
            output_dir=req.output_dir,
        )
        _jobs[run_id]["status"] = "completed"
        _jobs[run_id]["stats"] = stats
        _jobs[run_id]["finished_at"] = datetime.now(UTC).isoformat()
        logger.info("Ingest job %s completed", run_id)
    except Exception as exc:
        logger.exception("Ingest job %s failed: %s", run_id, exc)
        _jobs[run_id]["status"] = "failed"
        _jobs[run_id]["error"] = str(exc)
        _jobs[run_id]["finished_at"] = datetime.now(UTC).isoformat()


@app.post("/ingest", status_code=202)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Enqueue a graphgen KnowledgePipeline run (non-blocking)."""
    run_id = str(uuid.uuid4())
    queued_at = datetime.now(UTC).isoformat()
    _jobs[run_id] = {"status": "queued", "queued_at": queued_at}
    _ingest_task = asyncio.create_task(_run_ingest_task(run_id, req))
    logger.debug("Ingest task %s created: %r", run_id, _ingest_task)
    return IngestResponse(run_id=run_id, status="queued", queued_at=queued_at)


@app.get("/jobs/{run_id}", responses=_R404)
async def get_job(run_id: str) -> dict[str, Any]:
    """Poll ingest job status."""
    job = _jobs.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {run_id!r} not found")
    return {"run_id": run_id, **job}


# ── Retrieve ───────────────────────────────────────────────────────────────


@app.post("/retrieve", responses=_R503)
async def retrieve(req: RetrieveRequest) -> str:
    """Hybrid entity-first retrieval — returns XML string."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    ctx = await _retriever.retrieve(
        query=req.query,
        session_id=req.session_id,
        k_entity=req.k_entity,
        k_semantic=req.k_semantic,
        k_final=req.k_final,
    )
    return ctx.to_xml()


# ── Session lifecycle ──────────────────────────────────────────────────────


@app.post("/session")
async def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    """Create a new session namespace, optionally pre-seeding it from the corpus."""
    session_id = req.session_id or str(uuid.uuid4())
    seeded = 0
    if req.seed_entities and _neo is not None:
        seeded = await _neo.seed_session_from_corpus(session_id, req.seed_entities)
    logger.info("Session %s created (seeded=%d entities)", session_id, seeded)  # NOSONAR
    return SessionCreateResponse(session_id=session_id, seeded=seeded)


@app.delete("/session/{session_id}", responses=_R503)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    """Merge Memgraph session graph → Neo4j corpus, then purge session data."""
    if _memgraph is None or _neo is None or _pg is None:
        raise HTTPException(status_code=503, detail=_ERR_STORES_NOT_INIT)

    # Drain Memgraph session graph
    try:
        nodes = await _memgraph.get_session_nodes(session_id)
        edges = await _memgraph.get_session_edges(session_id)
    except Exception as exc:
        logger.warning("Memgraph read failed for session %s: %s", session_id, exc)  # NOSONAR
        nodes, edges = [], []

    # Promote session entities + relations into the corpus
    merged_nodes, merged_edges = await _neo.merge_session_entities_to_corpus(
        session_id, nodes, edges
    )

    # Purge session data from all stores
    await asyncio.gather(
        _memgraph.delete_session(session_id),
        _pg.delete_session(session_id),
    )

    logger.info(  # NOSONAR
        "Session %s merged: nodes=%d edges=%d", session_id, merged_nodes, merged_edges
    )
    return SessionDeleteResponse(
        session_id=session_id,
        merged_nodes=merged_nodes,
        merged_edges=merged_edges,
        purged_chunks=0,  # pg.delete_session returns count; we ignore here
    )


# ── Stats ──────────────────────────────────────────────────────────────────


@app.get("/stats", responses=_R503)
async def stats() -> dict[str, Any]:
    """Combined Neo4j node counts + pgvector row counts."""
    if _neo is None or _pg is None:
        raise HTTPException(status_code=503, detail=_ERR_STORES_NOT_INIT)

    neo_counts = await _neo.count_nodes()

    pool = _pg._pool  # noqa: SLF001 — internal access for stats only
    pg_counts: dict[str, int] = {"chunks": 0, "entities": 0}
    if pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT (SELECT count(*) FROM chunks) AS chunks, "
                "(SELECT count(*) FROM entities) AS entities"
            )
            if rows:
                pg_counts = {"chunks": int(rows[0]["chunks"]), "entities": int(rows[0]["entities"])}

    return {
        "neo4j": neo_counts,
        "pgvector": pg_counts,
    }


# ── Session preview-merge (E5) ─────────────────────────────────────────────


@app.get("/session/{session_id}/preview-merge", responses=_R503)
async def preview_session_merge(session_id: str) -> dict[str, Any]:
    """Preview what merging *session_id* into the corpus would do — no writes."""
    if _memgraph is None or _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_STORES_NOT_INIT)
    try:
        nodes = await _memgraph.get_session_nodes(session_id)
        edges = await _memgraph.get_session_edges(session_id)
        node_count = await _memgraph.get_session_node_count(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Memgraph unavailable: {exc}") from exc
    preview = await _neo.preview_session_merge(session_id, nodes, edges)
    preview["memgraph_node_count"] = node_count
    return preview


# ── Admin: Ontology Patch (E4) ─────────────────────────────────────────────


def _load_patch_taxonomy(taxonomy_file: str | None) -> dict[str, Any] | None:
    """Load taxonomy for a patch request; returns None on failure."""
    try:
        from graphknows.graphgen.config.taxonomy import (  # noqa: PLC0415
            TaxonomyConfig,
            get_default_taxonomy,
            load_taxonomy,
        )

        result: TaxonomyConfig = (
            load_taxonomy(taxonomy_file) if taxonomy_file else get_default_taxonomy()
        )
        return result.model_dump()
    except Exception as exc:
        logger.warning("Could not load patch taxonomy: %s", exc)
        return None


@app.post("/admin/patch", responses=_R503_500)
async def run_patch(req: PatchRequest) -> dict[str, Any]:
    """Run ontology patch phases on the Neo4j corpus."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialised")  # NOSONAR
    taxonomy = _load_patch_taxonomy(req.taxonomy_file)
    try:
        from graphknows.graphgen.patch.runner import run_patch as _run_patch

        result = await asyncio.to_thread(
            _run_patch,
            neo4j_uri=_settings.neo4j_uri,
            neo4j_user=_settings.neo4j_user,
            neo4j_password=_settings.neo4j_password,
            phases=req.phases,
            dry_run=req.dry_run,
            batch_size=req.batch_size,
            taxonomy=taxonomy,
        )
        return result
    except Exception as exc:
        logger.exception("Patch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc  # NOSONAR


async def preview_patch(
    phases: str | None = None,
    taxonomy_file: str | None = None,
) -> dict[str, Any]:
    """Dry-run all (or selected) patch phases and return the preview report."""
    phase_list = phases.split(",") if phases else None
    req = PatchRequest(phases=phase_list, dry_run=True, taxonomy_file=taxonomy_file)
    return await run_patch(req)


# ── Temporal SEGMENT endpoints (E6) ───────────────────────────────────────


@app.get("/segments", responses=_R503)
async def list_segments(limit: int = 20) -> list[dict[str, Any]]:
    """List SEGMENT provenance nodes ordered by ingestion time (newest first)."""
    if _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_NEO4J_NOT_INIT)
    return await _neo.list_segments(limit=limit)


@app.get("/segments/{segment_id}", responses=_R503_404)
async def get_segment(segment_id: str) -> dict[str, Any]:
    """Provenance + doc-count statistics for a single SEGMENT node."""
    if _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_NEO4J_NOT_INIT)
    info = await _neo.get_segment_info(segment_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id!r} not found")
    return info


# ── Config ────────────────────────────────────────────────────────────────


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Expose runtime configuration so agents can adapt to the deployment."""
    cfg = _settings or GraphKnowsSettings()
    return {
        "embed_model": cfg.embed_model,
        "vector_dims": 384,
        "rrf_k": 60,
        "session_ttl_minutes": cfg.session_ttl_minutes,
        "env": cfg.graphknows_env,
        "mcp_transport": "http",
    }


# ── Entity discovery & graph traversal ────────────────────────────────────


@app.get("/entities/search", responses=_R503)
async def search_entities(
    q: str,
    k: int = 10,
    types: str | None = None,
) -> list[dict[str, Any]]:
    """Discover corpus entities by semantic similarity.

    Args:
        q: Natural-language query or entity name.
        k: Maximum results (default 10).
        types: Comma-separated label filter, e.g. ``ORG,PERSON``.
    """
    if _retriever is None or _pg is None:
        raise HTTPException(status_code=503, detail=_ERR_STORES_NOT_INIT)
    from graphknows.retrieval.stores.embedder import embed

    entity_types = [t.strip() for t in types.split(",")] if types else None
    emb_array = await asyncio.to_thread(embed, [q], _settings.embed_model)
    emb = emb_array[0].tolist()
    return await _pg.search_entities(emb, k=k, entity_types=entity_types)


@app.get("/entities/{name}/neighbors", responses=_R503)
async def get_entity_neighbors(
    name: str,
    depth: int = 1,
) -> dict[str, Any]:
    """Return neighboring nodes reachable from *name* within *depth* hops."""
    if _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_NEO4J_NOT_INIT)
    return await _neo.get_entity_neighbors(name, depth=depth)


@app.get("/entities/{name}/community", responses=_R503_404)
async def get_entity_community(name: str) -> dict[str, Any]:
    """Return the Leiden community topic summary for a named entity."""
    if _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_NEO4J_NOT_INIT)
    topic = await _neo.get_entity_community(name)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"No community found for entity: {name!r}")
    return topic


# ── Chunk detail ───────────────────────────────────────────────────────────


@app.get("/chunks/{chunk_id}", responses=_R503_404)
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    """Return full chunk text, metadata, and linked entities."""
    if _pg is None or _neo is None:
        raise HTTPException(status_code=503, detail=_ERR_STORES_NOT_INIT)
    text_map = await _pg.get_chunks_by_ids([chunk_id])
    if chunk_id not in text_map:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id!r} not found")
    entities = await _neo.get_chunk_entities(chunk_id)
    return {**text_map[chunk_id], "chunk_id": chunk_id, "entities": entities}


# ── Unified search (JSON) ─────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., min_length=1)
    search_type: str = Field(default="HYBRID")
    session_id: str = Field(default="default")
    k_entity: int = Field(default=10, ge=1, le=100)
    k_semantic: int = Field(default=20, ge=1, le=200)
    k_final: int = Field(default=10, ge=1, le=100)


@app.post("/search", responses=_R503_422)
async def search(req: SearchRequest) -> dict[str, Any]:
    """Unified multi-mode search endpoint returning JSON.

    Equivalent to the MCP ``retrieve`` tool but returns a plain dict
    instead of XML, suitable for direct REST consumption.

    Search modes: HYBRID, SEMANTIC, GRAPH_TRAVERSAL, SUMMARIES, FEELING_LUCKY.
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")  # NOSONAR

    mode = req.search_type.upper()
    try:
        if mode == "SEMANTIC":
            ctx = await _retriever.retrieve_semantic(req.query, req.session_id, k=req.k_semantic)
        elif mode == "GRAPH_TRAVERSAL":
            ctx = await _retriever.retrieve_graph_traversal(
                req.query,
                req.session_id,
                k_entity=req.k_entity,
                depth=2,
                k_final=req.k_final,
            )
        elif mode == "SUMMARIES":
            ctx = await _retriever.retrieve_summaries(
                req.query, req.session_id, k_entity=req.k_entity
            )
        elif mode == "FEELING_LUCKY":
            ctx = await _retriever.retrieve_feeling_lucky(
                req.query,
                req.session_id,
                k_entity=req.k_entity,
                k_semantic=req.k_semantic,
                k_final=req.k_final,
            )
        elif mode == "HYBRID":
            ctx = await _retriever.retrieve(
                req.query,
                req.session_id,
                k_entity=req.k_entity,
                k_semantic=req.k_semantic,
                k_final=req.k_final,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown search_type {req.search_type!r}. "
                "Valid: HYBRID, SEMANTIC, GRAPH_TRAVERSAL, SUMMARIES, FEELING_LUCKY.",
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ctx.to_dict()
