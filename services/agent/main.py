"""
Graphknows Agent Service — session-aware FastAPI entry point.

Serves as the SOLE frontend entry point, wiring together:
  - SessionGraphManager (Memgraph ephemeral session graphs)
  - GraphRAG REST API (http://graphrag:8001/retrieve)

Endpoints
---------
POST   /chat                     — send a message, get an answer + live graph
GET    /session/{session_id}/graph — return current session subgraph (D3.js)
DELETE /session/{session_id}     — close/destroy a session
GET    /health                   — liveness probe
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from services.agent.session_graph import SessionGraphManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("agent.main")

# ---------------------------------------------------------------------------
# Config — all values from environment variables, with sane defaults
# ---------------------------------------------------------------------------

_KNOWN_ENV_KEYS = {
    "GRAPHRAG_URL",
    "MEMGRAPH_URI",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "HTTPX_TIMEOUT",
}


def _load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "graphrag_url": os.environ.get("GRAPHRAG_URL", "http://localhost:8001"),
        "memgraph_uri": os.environ.get("MEMGRAPH_URI", "bolt://172.17.0.1:17687"),
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        "httpx_timeout": float(os.environ.get("HTTPX_TIMEOUT", "60")),
    }

    # Warn on unknown AGENT_* / unexpected env keys that look like config
    for key in os.environ:
        if key.startswith("AGENT_") and key not in _KNOWN_ENV_KEYS:
            warnings.warn(
                f"Unknown agent config env var: {key!r} — ignoring.",
                stacklevel=1,
            )

    return cfg


CONFIG: dict[str, Any] = _load_config()

# ---------------------------------------------------------------------------
# Module-level singletons (initialised in lifespan)
# ---------------------------------------------------------------------------

_sgm = None          # SessionGraphManager instance
_http_client = None  # shared httpx.AsyncClient

# In-memory session registry — maps session_id → True (session is active)
_active_sessions: set[str] = set()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sgm, _http_client

    log.info("Starting Graphknows Agent service …")
    log.info("  graphrag_url  = %s", CONFIG["graphrag_url"])
    log.info("  memgraph_uri  = %s", CONFIG["memgraph_uri"])
    log.info("  neo4j_uri     = %s", CONFIG["neo4j_uri"])

    _sgm = SessionGraphManager(
        memgraph_uri=CONFIG["memgraph_uri"],
        neo4j_uri=CONFIG["neo4j_uri"],
        neo4j_user=CONFIG["neo4j_user"],
        neo4j_password=CONFIG["neo4j_password"],
    )

    _http_client = httpx.AsyncClient(
        base_url=CONFIG["graphrag_url"],
        timeout=CONFIG["httpx_timeout"],
    )

    log.info("Agent service ready.")
    yield

    # Shutdown
    log.info("Shutting down Agent service …")
    if _http_client:
        await _http_client.aclose()
    if _sgm:
        _sgm.close()
    log.info("Agent service stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Graphknows Agent",
    version="1.0.0",
    description=(
        "Session-aware agent entry point — wires GraphRAG retrieval with "
        "Memgraph ephemeral session graphs for political intelligence."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_message(self) -> "ChatRequest":
        if not self.message.strip():
            raise ValueError("message must not be empty")
        return self


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    graph: GraphResponse


# ---------------------------------------------------------------------------
# Helper: call graphrag /retrieve
# ---------------------------------------------------------------------------


async def _call_graphrag(message: str, session_id: str) -> dict:
    """POST to graphrag service and return the parsed JSON body."""
    payload = {
        "query": message,
        "session_id": session_id,
        "max_steps": 8,
    }
    try:
        resp = await _http_client.post("/retrieve", json=payload)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("graphrag returned HTTP %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"graphrag error {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        log.error("graphrag connection error: %s", exc)
        raise HTTPException(status_code=502, detail=f"graphrag unreachable: {exc}") from exc

    return resp.json()


# ---------------------------------------------------------------------------
# Helper: extract entities from graphrag response and upsert into Memgraph
# ---------------------------------------------------------------------------


def _upsert_entities(session_id: str, graphrag_response: dict) -> None:
    """
    Extract entities_mentioned (and citation entities) from the graphrag
    response and upsert them into the Memgraph session graph.
    Runs synchronously (called via asyncio.to_thread).
    """
    # graphrag /retrieve returns citations list; each citation may have an
    # 'entity' key.  The RetrieveResponse also surfaces entities_mentioned
    # directly when present (future-proofing).
    entities: list[dict] = []

    # Flat list under 'entities_mentioned' if graphrag surfaces it
    for name in graphrag_response.get("entities_mentioned", []):
        if isinstance(name, str):
            entities.append({"name": name, "label": "Entity"})
        elif isinstance(name, dict):
            entities.append(name)

    # Pull entity names from citations
    for citation in graphrag_response.get("citations", []):
        if isinstance(citation, dict):
            entity_name = citation.get("entity") or citation.get("name")
            if entity_name and isinstance(entity_name, str):
                entities.append({"name": entity_name, "label": "Entity"})

    if not entities:
        log.debug("No entities to upsert for session %s", session_id)
        return

    # De-duplicate by name
    seen: set[str] = set()
    unique_entities = []
    for e in entities:
        n = e.get("name", "")
        if n and n not in seen:
            seen.add(n)
            unique_entities.append(e)

    log.info("Upserting %d entities for session %s", len(unique_entities), session_id)
    _sgm.seed_from_neo4j(session_id, [e["name"] for e in unique_entities])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "graphknows-agent"}


@app.post("/chat", response_model=ChatResponse, summary="Send a message to the agent")
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Create or continue a session, call graphrag /retrieve, upsert returned
    entities into the Memgraph session graph, and return the agent answer
    together with the current session subgraph for D3.js rendering.
    """
    # Resolve or create session
    session_id: str = req.session_id or str(uuid.uuid4())
    is_new = session_id not in _active_sessions

    if is_new:
        log.info("Creating new session: %s", session_id)
        await asyncio.to_thread(_sgm.create_session, session_id)
        _active_sessions.add(session_id)
    else:
        log.info("Continuing session: %s", session_id)

    # Call graphrag
    graphrag_resp = await _call_graphrag(req.message, session_id)

    # Upsert entities into Memgraph (IO-bound but uses the neo4j driver sync API)
    await asyncio.to_thread(_upsert_entities, session_id, graphrag_resp)

    # Fetch the live subgraph
    subgraph: dict = await asyncio.to_thread(_sgm.get_session_graph, session_id)

    answer: str = graphrag_resp.get("answer", "")

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        graph=GraphResponse(
            nodes=subgraph.get("nodes", []),
            edges=subgraph.get("edges", []),
        ),
    )


@app.get(
    "/session/{session_id}/graph",
    response_model=GraphResponse,
    summary="Get the current session subgraph",
)
async def get_session_graph(session_id: str) -> GraphResponse:
    """Return the D3.js-ready subgraph for the given session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    subgraph: dict = await asyncio.to_thread(_sgm.get_session_graph, session_id)
    return GraphResponse(
        nodes=subgraph.get("nodes", []),
        edges=subgraph.get("edges", []),
    )


@app.delete("/session/{session_id}", summary="Close and destroy a session")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Close the session and remove all ephemeral data from Memgraph."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    await asyncio.to_thread(_sgm.close_session, session_id)
    _active_sessions.discard(session_id)
    log.info("Session deleted: %s", session_id)
    return {"status": "deleted", "session_id": session_id}
