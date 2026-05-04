"""GraphRAG FastAPI REST wrapper.

Exposes the GraphRAGAgent ReAct agent as an async HTTP service.

Endpoints
---------
GET  /health     — liveness probe
POST /retrieve   — run the ReAct agent and return structured response

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

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure the graphrag package directory is on sys.path so relative imports work
_SVC_DIR = Path(__file__).parent
if str(_SVC_DIR) not in sys.path:
    sys.path.insert(0, str(_SVC_DIR))

REPO = Path(__file__).parent.parent.parent
log = logging.getLogger("graphrag.api")


def _build_cfg() -> dict:
    """Load environment and build the agent config dict."""
    env = dotenv_values(str(REPO / ".env"))

    neo4j_uri = env.get("NEO4J_URI") or (
        f"bolt://{env.get('NEO4J_HOST', '172.18.0.1')}:{env.get('NEO4J_PORT', 7687)}"
    )
    pg_user = env.get("PGVECTOR_USER") or env.get("PGUSER", "postgres")
    pg_pass = env.get("PGVECTOR_PASSWORD") or env.get("PGPASSWORD", "pgvectorpass123")
    pg_host = env.get("PGVECTOR_HOST") or env.get("PGHOST", "172.18.0.1")
    pg_port = env.get("PGVECTOR_PORT") or env.get("PGPORT", 5432)
    pg_db = env.get("PGVECTOR_DB") or env.get("PGDATABASE", "vectordb")
    pgvector_dsn = (
        env.get("PGVECTOR_DSN")
or f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    )

    from graph_store import EMBED_MODEL

    return {
        "neo4j_uri": neo4j_uri,
        "neo4j_user": env.get("NEO4J_USER", "neo4j"),
        "neo4j_password": env.get("NEO4J_PASSWORD", "testpassword123"),
        "neo4j_database": env.get("NEO4J_DATABASE", "neo4j"),
        "pgvector_dsn": pgvector_dsn,
        "embed_model": env.get("EMBED_MODEL", EMBED_MODEL),
        "groq_api_key": env.get("GROQ_API_KEY") or env.get("LLM_GROQ_API_KEY", ""),
        "llm_model": env.get("LLM_MODEL", "llama-3.3-70b-versatile"),
    }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve."""

    query: str
    max_steps: int = 8
    session_id: Optional[str] = None  # reserved for future Memgraph session


class RetrieveResponse(BaseModel):
    """Structured response from the GraphRAG ReAct agent."""

    query: str
    answer: str
    citations: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    n_steps: int
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan: instantiate agent once at startup
# ---------------------------------------------------------------------------

_agent = None  # module-level holder so endpoints can reference it


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    log.info("Initialising GraphRAGAgent …")
    from agent import GraphRAGAgent

    cfg = _build_cfg()
    _agent = GraphRAGAgent(cfg)
    log.info("GraphRAGAgent ready.")
    yield
    # Shutdown
    if _agent is not None:
        _agent.close()
        log.info("GraphRAGAgent closed.")


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


@app.post("/retrieve", response_model=RetrieveResponse, summary="Run the ReAct agent")
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Run the GraphRAG ReAct agent for the given query.

    The agent is NOT async-native; it is executed in a thread pool via
    ``asyncio.to_thread`` so the event loop is never blocked.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialised")

    try:
        result: dict = await asyncio.to_thread(_agent.run, req.query, req.max_steps)
    except Exception as exc:
        log.exception("Agent raised an exception for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # The agent returns: query, answer, trace, citations, steps
    # Map to the canonical response shape expected by this API.
    return RetrieveResponse(
        query=result.get("query", req.query),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        reasoning_steps=result.get("trace", []),
        n_steps=result.get("steps", 0),
        session_id=req.session_id,
    )
