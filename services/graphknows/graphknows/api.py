"""graphknows — unified agentic memory service.

Endpoints
---------
GET  /health                 — liveness probe
POST /ingest                 — enqueue graphgen KnowledgePipeline (background task)
GET  /jobs/{run_id}          — poll ingest job status
POST /retrieve               — hybrid entity-graph + semantic retrieval → XML
POST /session                — create or re-seed a session namespace
DELETE /session/{session_id} — merge Memgraph session → Neo4j corpus and purge
GET  /stats                  — combined Neo4j + pgvector row counts
MCP  /mcp/                   — FastMCP tool endpoint (retrieve, community_summary)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from graphknows.mcp.server import mcp, set_dependencies
from graphknows.retrieval.retriever import HybridRetriever
from graphknows.stores.memgraph import MemgraphStore
from graphknows.stores.neo4j import Neo4jStore
from graphknows.stores.pgvector import PgVectorStore
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_handler = logging.StreamHandler()
_handler.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("graphknows.api")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class GraphKnowsSettings(BaseSettings):
    """Runtime configuration for the graphknows service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    graphknows_env: str = "development"
    graphknows_port: int = 8025

    # Databases
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jpassword"
    neo4j_database: str = "neo4j"

    pgvector_dsn: str = "postgresql://graphuser:graphpass@pgvector:5432/graphdb"

    memgraph_uri: str = "bolt://memgraph:7687"

    # Embedding model
    embed_model: str = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


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
async def _lifespan(app: FastAPI):
    global _pg, _neo, _memgraph, _retriever, _settings

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
        set_dependencies(_retriever, _pg, _neo, {"embed_model": _settings.embed_model})
        logger.info("graphknows ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("graphknows started in degraded mode: %s", exc)

    yield

    logger.info("graphknows shutting down")
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

    id: str | None = Field(default=None, description="Optional document ID (auto-generated if omitted)")
    text: str = Field(..., description="Document body text")
    title: str | None = Field(default=None, description="Optional document title")
    source: str | None = Field(default=None, description="Optional source URL or label")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    documents: list[DocumentIn] = Field(..., min_length=1)
    config_override: dict[str, Any] | None = Field(default=None)
    output_dir: str | None = Field(default=None)


class IngestResponse(BaseModel):
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
    session_id: str
    seeded: int


class SessionDeleteResponse(BaseModel):
    session_id: str
    merged_nodes: int
    merged_edges: int
    purged_chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "graphknows"}


# ── Ingest ─────────────────────────────────────────────────────────────────


async def _run_ingest_task(run_id: str, req: IngestRequest) -> None:
    """Background task: run graphgen pipeline and update job store."""
    _jobs[run_id]["status"] = "running"
    _jobs[run_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        from graphknows.ingestion.pipeline import run_ingestion

        docs = [d.model_dump() for d in req.documents]
        stats = await run_ingestion(
            documents=docs,
            config_override=req.config_override,
            output_dir=req.output_dir,
        )
        _jobs[run_id]["status"] = "completed"
        _jobs[run_id]["stats"] = stats
        _jobs[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Ingest job %s completed", run_id)
    except Exception as exc:
        logger.exception("Ingest job %s failed: %s", run_id, exc)
        _jobs[run_id]["status"] = "failed"
        _jobs[run_id]["error"] = str(exc)
        _jobs[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Enqueue a graphgen KnowledgePipeline run (non-blocking)."""
    run_id = str(uuid.uuid4())
    queued_at = datetime.now(timezone.utc).isoformat()
    _jobs[run_id] = {"status": "queued", "queued_at": queued_at}
    asyncio.create_task(_run_ingest_task(run_id, req))
    return IngestResponse(run_id=run_id, status="queued", queued_at=queued_at)


@app.get("/jobs/{run_id}")
async def get_job(run_id: str) -> dict[str, Any]:
    """Poll ingest job status."""
    job = _jobs.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {run_id!r} not found")
    return {"run_id": run_id, **job}


# ── Retrieve ───────────────────────────────────────────────────────────────


@app.post("/retrieve")
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


@app.post("/session", response_model=SessionCreateResponse)
async def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    """Create a new session namespace, optionally pre-seeding it from the corpus."""
    session_id = req.session_id or str(uuid.uuid4())
    seeded = 0
    if req.seed_entities and _neo is not None:
        seeded = await _neo.seed_session_from_corpus(session_id, req.seed_entities)
    logger.info("Session %s created (seeded=%d entities)", session_id, seeded)
    return SessionCreateResponse(session_id=session_id, seeded=seeded)


@app.delete("/session/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    """Merge Memgraph session graph → Neo4j corpus, then purge session data."""
    if _memgraph is None or _neo is None or _pg is None:
        raise HTTPException(status_code=503, detail="Stores not initialised")

    # Drain Memgraph session graph
    try:
        nodes = await _memgraph.get_session_nodes(session_id)
        edges = await _memgraph.get_session_edges(session_id)
    except Exception as exc:
        logger.warning("Memgraph read failed for session %s: %s", session_id, exc)
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

    logger.info(
        "Session %s merged: nodes=%d edges=%d", session_id, merged_nodes, merged_edges
    )
    return SessionDeleteResponse(
        session_id=session_id,
        merged_nodes=merged_nodes,
        merged_edges=merged_edges,
        purged_chunks=0,  # pg.delete_session returns count; we ignore here
    )


# ── Stats ──────────────────────────────────────────────────────────────────


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Combined Neo4j node counts + pgvector row counts."""
    if _neo is None or _pg is None:
        raise HTTPException(status_code=503, detail="Stores not initialised")

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
