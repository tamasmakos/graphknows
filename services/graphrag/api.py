"""GraphRAG FastAPI REST wrapper.

Exposes the HybridRetriever as a pure retrieval service (no LLM, no ReAct).

Endpoints
---------
GET  /health     — liveness probe
POST /retrieve   — hybrid retrieval (RRF fusion of chunks + entities)
POST /path       — shortest path between two entities in the knowledge graph
POST /chunks     — source chunks for a given entity
GET  /community/{entity_id} — Leiden community lookup for an entity

Usage (development)::

    uvicorn graphrag.api:app --port 8097 --reload

Or from the repo root::

    PYTHONPATH=/root/graphknows/services \
        uvicorn graphrag.api:app --port 8097 --reload
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphknows.shared.config import BaseServiceSettings

# Ensure the graphrag package directory is on sys.path so relative imports work
_SVC_DIR = Path(__file__).parent
if str(_SVC_DIR) not in sys.path:
    sys.path.insert(0, str(_SVC_DIR))

log = logging.getLogger("graphrag.api")


class GraphRAGSettings(BaseServiceSettings):
    """Settings for the GraphRAG retrieval service.

    Inherits all database and embedding config from BaseServiceSettings.
    """


def _build_cfg() -> dict:
    """Build config dict from GraphRAGSettings Pydantic model."""
    settings = GraphRAGSettings()
    return {
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_user": settings.neo4j_user,
        "neo4j_password": settings.neo4j_password,
        "neo4j_database": settings.neo4j_database,
        "pgvector_dsn": settings.pgvector_dsn,
        "embed_model": settings.embed_model,
        "groq_api_key": settings.groq_api_key,
        "llm_model": settings.llm_model,
    }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve."""

    query: str
    k: int = 10


class RetrievalResult(BaseModel):
    """Pure retrieval result — no LLM synthesis."""

    chunks: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]


class PathRequest(BaseModel):
    """Request body for POST /path."""

    entity_a: str
    entity_b: str
    max_hops: int = 3


class PathResult(BaseModel):
    """Shortest path result between two entities."""

    path: List[Dict[str, Any]]
    found: bool


class ChunksRequest(BaseModel):
    """Request body for POST /chunks."""

    entity_id: str
    top_k: int = 5


class ChunksResult(BaseModel):
    """Source chunks for an entity."""

    chunks: List[Dict[str, Any]]


class CommunityResult(BaseModel):
    """Leiden community lookup result."""

    community_id: Optional[int] = None
    label: Optional[str] = None
    members: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Lifespan: instantiate retriever once at startup
# ---------------------------------------------------------------------------

_retriever = None  # module-level holder so endpoints can reference it


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    log.info("Initialising HybridRetriever …")
    from retriever import HybridRetriever

    cfg = _build_cfg()
    _retriever = HybridRetriever(cfg)
    log.info("HybridRetriever ready.")
    yield
    # Shutdown
    if _retriever is not None:
        _retriever.close()
        log.info("HybridRetriever closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Graphknows GraphRAG",
    version="1.0.0",
    description="REST API for the GraphRAG ReAct agent — political intelligence over EU Parliament debates.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe")
async def health() -> Dict[str, str]:
    """Return a simple OK payload so load-balancers and k8s probes work."""
    return {"status": "ok", "service": "graphknows-graphrag"}


@app.post("/retrieve", response_model=RetrievalResult, summary="Hybrid retrieval")
async def retrieve(req: RetrieveRequest) -> RetrievalResult:
    """Run hybrid retrieval (RRF fusion of chunks + entities).

    No LLM call — pure retrieval from pgvector and Neo4j.
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialised")

    try:
        result: dict = await asyncio.to_thread(
            _retriever.retrieve, req.query, req.k
        )
    except Exception as exc:
        log.exception("Retriever raised an exception for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RetrievalResult(
        chunks=result.get("chunks", []),
        entities=result.get("entities", []),
        scores=result.get("scores", []),
    )


@app.post("/path", response_model=PathResult, summary="Shortest path between entities")
async def get_path(req: PathRequest) -> PathResult:
    """Find shortest path between two entities over LKIF_REL in Neo4j."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialised")

    try:
        result: dict = await asyncio.to_thread(
            _retriever.get_path, req.entity_a, req.entity_b, req.max_hops
        )
    except Exception as exc:
        log.exception("Path lookup failed for %r -> %r", req.entity_a, req.entity_b)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PathResult(path=result.get("path", []), found=result.get("found", False))


@app.post("/chunks", response_model=ChunksResult, summary="Chunks for entity")
async def get_chunks(req: ChunksRequest) -> ChunksResult:
    """Return source chunks that mention a given entity."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialised")

    try:
        result: list = await asyncio.to_thread(
            _retriever.get_chunks, req.entity_id, req.top_k
        )
    except Exception as exc:
        log.exception("Chunk lookup failed for entity %r", req.entity_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChunksResult(chunks=result)


@app.get("/community/{entity_id}", response_model=CommunityResult, summary="Community lookup")
async def get_community(entity_id: str) -> CommunityResult:
    """Return Leiden community info for an entity."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialised")

    try:
        result: dict = await asyncio.to_thread(_retriever.get_community, entity_id)
    except Exception as exc:
        log.exception("Community lookup failed for entity %r", entity_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CommunityResult(
        community_id=result.get("community_id"),
        label=result.get("label"),
        members=result.get("members", []),
    )
