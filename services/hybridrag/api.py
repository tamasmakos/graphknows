"""HybridRAG FastAPI service.

Endpoints
---------
GET  /health                — liveness probe
POST /retrieve              — hybrid retrieval → XML string
GET  /community/{entity_id} — Leiden community lookup
GET  /stats                 — Neo4j + pgvector node counts

MCP is mounted at /mcp/ (Streamable HTTP transport).
Ingest is handled by graphgen (the ingestion service).

Usage (development)::

    uvicorn hybridrag.api:app --port 8020 --reload
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger


_SHARED_PKG_DIR = (
    Path(__file__).resolve().parents[2] / "libs" / "python" / "graphknows-shared"
)
if str(_SHARED_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_PKG_DIR))

from graphknows.shared.config import BaseServiceSettings

log = logging.getLogger("hybridrag.api")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class HybridRAGSettings(BaseServiceSettings):
    """Settings for the HybridRAG retrieval service."""

    hybridrag_port: int = 8020


def _build_cfg() -> dict:
    s = HybridRAGSettings()
    return {
        "neo4j_uri": s.neo4j_uri,
        "neo4j_user": s.neo4j_user,
        "neo4j_password": s.neo4j_password,
        "neo4j_database": s.neo4j_database,
        "pgvector_dsn": s.pgvector_dsn,
        "embed_model": s.embed_model,
        "groq_api_key": s.groq_api_key,
        "llm_model": s.llm_model,
    }


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str
    session_id: str
    k_entity: int = 10
    k_semantic: int = 20
    k_final: int = 10


class CommunityResult(BaseModel):
    id: str | None = None
    label: str | None = None
    summary: str | None = None


class StatsResult(BaseModel):
    neo4j: dict[str, int]


# ---------------------------------------------------------------------------
# Lifespan: initialise all stores once at startup
# ---------------------------------------------------------------------------

_pg = None
_neo = None
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    _configure_logging()
    global _pg, _neo, _retriever

    log.info("HybridRAG startup …")
    cfg = _build_cfg()

    from hybridrag.stores.neo4j import Neo4jStore
    from hybridrag.stores.pgvector import PgVectorStore
    from hybridrag.retriever import HybridRetriever

    _pg = PgVectorStore(cfg["pgvector_dsn"])
    await _pg.connect()
    await _pg.ensure_schema()

    _neo = Neo4jStore(
        cfg["neo4j_uri"],
        cfg["neo4j_user"],
        cfg["neo4j_password"],
        cfg["neo4j_database"],
    )
    await _neo.ensure_indexes()

    _retriever = HybridRetriever(_pg, _neo, embed_model=cfg["embed_model"])

    from hybridrag.mcp.server import set_dependencies
    set_dependencies(_retriever, _pg, _neo, cfg)

    log.info("HybridRAG ready — pgvector + Neo4j + MCP wired.")

    yield

    log.info("HybridRAG shutdown …")
    if _pg:
        await _pg.close()
    if _neo:
        _neo.close()


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HybridRAG",
    version="1.0.0",
    description=(
        "Entity-first hybrid retrieval: graph traversal + pgvector semantic search + RRF. "
        "Exposes REST endpoints and an MCP server at /mcp/."
    ),
    lifespan=lifespan,
)

# Mount the FastMCP streamable-HTTP transport
from hybridrag.mcp.server import mcp as _mcp_server  # noqa: E402

app.mount("/mcp", _mcp_server.streamable_http_app())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "hybridrag"}


@app.post("/retrieve", response_model=None)
async def retrieve(req: RetrieveRequest) -> str:
    """Run hybrid retrieval and return the XML context string."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    ctx = await _retriever.retrieve(
        req.query,
        req.session_id,
        k_entity=req.k_entity,
        k_semantic=req.k_semantic,
        k_final=req.k_final,
    )
    return ctx.to_xml()


@app.post("/ingest", response_model=None, include_in_schema=False)
async def ingest_removed() -> dict:
    """Ingest is handled by graphgen, not hybridrag."""
    raise HTTPException(
        status_code=410,
        detail="Ingest is handled by graphgen. POST to the graphgen /ingest endpoint.",
    )


@app.get("/community/{entity_id}", response_model=CommunityResult)
async def community(entity_id: str) -> CommunityResult:
    """Return Leiden community topic for a named entity."""
    if _neo is None:
        raise HTTPException(status_code=503, detail="Neo4j store not initialised")
    topic = await _neo.get_entity_community(entity_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"No community for entity: {entity_id}")
    return CommunityResult(
        id=topic["id"],
        label=topic["label"],
        summary=topic["summary"],
    )


@app.get("/stats", response_model=StatsResult)
async def stats() -> StatsResult:
    """Return node/edge counts from Neo4j."""
    if _neo is None:
        raise HTTPException(status_code=503, detail="Neo4j store not initialised")
    counts = await _neo.stats()
    return StatsResult(neo4j=counts)
