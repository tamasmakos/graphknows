"""Router — all HTTP + WS route handlers for the agent service."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator

from agent.config import AgentSettings
from agent.session.manager import SessionGraphManager
from agent.session_end import merge_session_to_neo4j
try:
    from graphknows.shared.models.session import StreamEvent
except Exception:
    StreamEvent = None  # type: ignore

log = logging.getLogger("agent.router")

router = APIRouter()

# Module-level singletons (set by lifespan in main.py)
_settings: AgentSettings | None = None
_sgm: SessionGraphManager | None = None
_http_client: httpx.AsyncClient | None = None
_ws_connections: dict[str, WebSocket] = {}


async def _broadcast_graph_update(session_id: str, payload: dict) -> None:
    ws = _ws_connections.get(session_id)
    if ws:
        try:
            await ws.send_text(json.dumps({"type": "graph_update", **payload}))
        except Exception:
            pass


def _get_settings() -> AgentSettings:
    if _settings is None:
        raise HTTPException(status_code=500, detail="Settings not initialized")
    return _settings


def _get_sgm() -> SessionGraphManager:
    if _sgm is None:
        raise HTTPException(status_code=500, detail="SessionGraphManager not initialized")
    return _sgm


def _get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    return _http_client


def _init_router(settings: AgentSettings, sgm: SessionGraphManager, http_client: httpx.AsyncClient) -> None:
    """Initialize router singletons. Called from main.py lifespan."""
    global _settings, _sgm, _http_client
    _settings = settings
    _sgm = sgm
    _http_client = http_client


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for POST /chat endpoint."""
    message: str
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_message(self) -> "ChatRequest":
        if not self.message.strip():
            raise ValueError("message must not be empty")
        return self


class GraphResponse(BaseModel):
    """Response model for graph endpoints."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    """Response model for POST /chat endpoint."""
    answer: str
    session_id: str
    graph: GraphResponse
    reasoning_steps: list = []
    sources: list = []


class QueryRequest(BaseModel):
    """Request body for POST /sessions/{id}/query endpoint."""
    message: str

    @model_validator(mode="after")
    def _check_message(self) -> "QueryRequest":
        if not self.message.strip():
            raise ValueError("message must not be empty")
        return self


# ---------------------------------------------------------------------------
# In-memory session registry
# ---------------------------------------------------------------------------

_active_sessions: set[str] = set()


# ---------------------------------------------------------------------------
# Helper: call graphrag /retrieve (for /chat backward compatibility)
# ---------------------------------------------------------------------------


async def _call_graphrag(message: str, session_id: str) -> dict:
    """POST to graphrag service and return the parsed JSON body.

    Args:
        message: The user query to send to graphrag.
        session_id: Session identifier for context tracking.

    Returns:
        Parsed JSON response from graphrag.

    Raises:
        HTTPException: If graphrag returns an error or is unreachable.
    """
    client = _get_http_client()
    payload = {
        "query": message,
        "session_id": session_id,
        "max_steps": 8,
    }
    try:
        resp = await client.post(f"{_get_settings().graphrag_url}/retrieve", json=payload)
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


def _infer_doc_metadata(doc_id: str) -> dict[str, str]:
    """Infer date, institution, and URL from a doc_id string.

    Args:
        doc_id: Document identifier, e.g. 'ep_debate_2024_03_13'.

    Returns:
        Dict with keys 'date', 'institution', 'url'.
    """
    import re as _re
    date = ""
    institution = ""
    url = ""
    if doc_id.startswith("ep_debate_"):
        institution = "European Parliament"
        url = "https://europarl.europa.eu/plenary/en/debates.html"
        m = _re.search(r"(\d{4})[_\-](\d{2})[_\-](\d{2})", doc_id)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return {"date": date, "institution": institution, "url": url}


def _extract_tool_name(action: str | None) -> str:
    """Extract tool name from an action string like 'search_entities(query=..., k=8)'.

    Args:
        action: Raw action string from the reasoning trace.

    Returns:
        The tool name, or empty string if not parseable.
    """
    import re as _re
    if not action:
        return ""
    m = _re.match(r"(\w+)\(", action.strip())
    return m.group(1) if m else ""


def _observation_type(tool_name: str) -> str:
    """Map a tool name to a canonical observation_type label.

    Args:
        tool_name: Tool name string.

    Returns:
        One of 'entities'|'chunks'|'community'|'neighbours'|'ranked'|'error'|'unknown'.
    """
    mapping = {
        "search_entities": "entities",
        "search_chunks": "chunks",
        "get_community": "community",
        "get_neighbours": "neighbours",
        "fuse_and_rank": "ranked",
        "get_chunks_for_entity": "chunks",
    }
    return mapping.get(tool_name, "unknown")


def _fetch_cooccurs(entity_names: list[str], limit: int = 50) -> list[tuple[str, str, int]]:
    """Fetch CO_OCCURS triplets for the given entity names from Neo4j.

    Args:
        entity_names: List of entity names to look up.
        limit: Maximum number of triplets to return.

    Returns:
        List of (subject, object, weight) tuples.
    """
    if not entity_names:
        return []
    try:
        from neo4j import GraphDatabase as _GraphDatabase
        settings = _get_settings()
        driver = _GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as neo_session:
            result = neo_session.run(
                """
                MATCH (a:Entity)-[r:CO_OCCURS]->(b:Entity)
                WHERE a.name IN $names OR b.name IN $names
                RETURN a.name AS subject, b.name AS object, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT $limit
                """,
                names=entity_names,
                limit=limit,
            )
            rows = [(r["subject"], r["object"], int(r["weight"] or 0)) for r in result]
        driver.close()
        return rows
    except Exception as exc:
        log.warning("_fetch_cooccurs failed: %s", exc)
        return []


def _upsert_entities(session_id: str, graphrag_response: dict) -> None:
    """Extract entities, RELATES_TO edges, Topic nodes, and Chunk nodes from the
    full graphrag agent trace and upsert them into the Memgraph session graph.

    Args:
        session_id: The active Memgraph session identifier.
        graphrag_response: The full JSON response from the graphrag /retrieve endpoint.
    """
    import re as _re

    entity_names: list[str] = []
    relates_to: list[tuple[str, str]] = []
    topic_nodes: list[dict] = []
    chunk_nodes: list[dict] = []

    sgm = _get_sgm()

    # ── Pull from explicit top-level fields ─────────────────────────────
    for name in graphrag_response.get("entities_mentioned", []):
        if isinstance(name, str):
            entity_names.append(name)
        elif isinstance(name, dict) and name.get("name"):
            entity_names.append(name["name"])

    for citation in graphrag_response.get("citations", []):
        if isinstance(citation, dict):
            cid = citation.get("chunk_id", "")
            doc_id = citation.get("doc_id", "")
            if cid:
                meta = _infer_doc_metadata(doc_id)
                chunk_nodes.append({
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "excerpt": citation.get("excerpt", "")[:300],
                    **meta,
                })

    # ── Parse ALL trace steps ────────────────────────────────────────────
    last_entity_list: list[str] = []

    for step in graphrag_response.get("reasoning_steps", []):
        action: str = step.get("action", "") or ""
        obs = step.get("observation")
        tool = _extract_tool_name(action)

        if obs is None:
            continue

        if tool in ("search_entities", "get_neighbours") and isinstance(obs, list):
            step_entities: list[str] = []
            for item in obs:
                if isinstance(item, dict) and item.get("name"):
                    step_entities.append(item["name"])
                    entity_names.append(item["name"])
            if tool == "get_neighbours" and last_entity_list and step_entities:
                for anchor in last_entity_list[:1]:
                    for neighbour in step_entities[:8]:
                        if anchor != neighbour:
                            relates_to.append((anchor, neighbour))
            if tool == "search_entities":
                last_entity_list = step_entities
            elif tool == "get_neighbours":
                for i, ea in enumerate(step_entities[:6]):
                    for eb in step_entities[i + 1:6]:
                        relates_to.append((ea, eb))

        elif tool == "get_community":
            items = obs if isinstance(obs, list) else [obs]
            m = _re.search(r'get_community\(["\']?([^"\'(),]+)["\']?\)', action)
            entity_name = m.group(1).strip() if m else ""
            for item in items:
                if isinstance(item, dict) and (item.get("id") or item.get("community_id")):
                    topic_nodes.append({**item, "entity_name": entity_name})

        elif tool in ("search_chunks", "fuse_and_rank", "get_chunks_for_entity") and isinstance(obs, list):
            for item in obs:
                if isinstance(item, dict):
                    cid = item.get("chunk_id", "")
                    if cid:
                        doc_id = item.get("doc_id", "")
                        meta = _infer_doc_metadata(doc_id)
                        chunk_nodes.append({
                            "chunk_id": cid,
                            "doc_id": doc_id,
                            "excerpt": item.get("text", "")[:300],
                            **meta,
                        })

    # ── Mine [BRACKETED] names from final answer text ────────────────────
    answer_text = graphrag_response.get("answer", "")
    for match in _re.findall(r'\[([A-Z][A-Z0-9_\s]+)\]', answer_text):
        clean = match.strip()
        if 2 < len(clean) < 60 and not clean.startswith("chunk"):
            entity_names.append(clean)

    if not entity_names and not topic_nodes and not chunk_nodes:
        log.debug("No graph enrichments to upsert for session %s", session_id)
        return

    # De-duplicate entities
    seen: set[str] = set()
    unique_entities: list[str] = []
    for n in entity_names:
        if n and n not in seen:
            seen.add(n)
            unique_entities.append(n)

    # ── Query Neo4j CO_OCCURS edges for found entities ───────────────────
    cooccurs_edges: list[tuple[str, str, int]] = _fetch_cooccurs(unique_entities)
    log.info(
        "Session %s CO_OCCURS from Neo4j: %d edges for %d entities",
        session_id, len(cooccurs_edges), len(unique_entities),
    )

    # De-duplicate chunk_nodes by chunk_id (last write wins)
    chunk_by_id: dict[str, dict] = {}
    for c in chunk_nodes:
        if c.get("chunk_id"):
            chunk_by_id[c["chunk_id"]] = c

    log.info(
        "Session %s enrichment: %d entities, %d relates_to, %d topics, %d chunks",
        session_id, len(unique_entities), len(relates_to), len(topic_nodes), len(chunk_by_id),
    )

    # Seed entities from Neo4j
    if unique_entities:
        sgm.seed_from_neo4j(session_id, unique_entities)

    # Enrich with richer graph structure
    sgm.add_graph_enrichments(
        session_id,
        relates_to=list(dict.fromkeys(relates_to)),
        topic_nodes=topic_nodes,
        chunk_nodes=list(chunk_by_id.values()),
    )

    # Upsert CO_OCCURS edges from Neo4j into the session graph
    if cooccurs_edges:
        sgm.add_cooccurs_edges(session_id, cooccurs_edges)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", summary="Liveness probe")
async def health() -> Dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "graphknows-agent"}


@router.post("/chat", response_model=ChatResponse, summary="Send a message to the agent")
async def chat(req: ChatRequest) -> ChatResponse:
    """HTTP fallback for /chat — routes through the LangGraph agent."""
    from agent.langgraph_agent import build_graph
    from langchain_core.messages import HumanMessage

    sgm = _get_sgm()
    settings = _get_settings()

    session_id: str = req.session_id or str(uuid.uuid4())
    is_new = session_id not in _active_sessions

    if is_new:
        log.info("Creating new session via HTTP: %s", session_id)
        await asyncio.to_thread(sgm.create_session, session_id)
        _active_sessions.add(session_id)

    graph = build_graph(
        ws_broadcaster=None,
        session_id=session_id,
        groq_api_key=settings.groq_api_key,
    )

    full_answer = ""
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=req.message)]},
        config={"recursion_limit": 12},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                full_answer += token

    subgraph: dict = await asyncio.to_thread(sgm.get_session_graph, session_id)

    return ChatResponse(
        answer=full_answer,
        session_id=session_id,
        graph=GraphResponse(
            nodes=subgraph.get("nodes", []),
            edges=subgraph.get("edges", []),
        ),
        reasoning_steps=[],
        sources=[],
    )


@router.post("/sessions", summary="Create a new session")
async def create_session() -> Dict[str, str]:
    """Create a new session and return the session ID."""
    sgm = _get_sgm()
    session_id = str(uuid.uuid4())
    await asyncio.to_thread(sgm.create_session, session_id)
    _active_sessions.add(session_id)
    log.info("Created session: %s", session_id)
    return {"session_id": session_id}


@router.get("/sessions/{session_id}", summary="Get session state")
async def get_session(session_id: str) -> Dict[str, Any]:
    """Return session state information.

    Args:
        session_id: The session identifier.

    Returns:
        Dict with session status and metadata.
    """
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return {"session_id": session_id, "status": "active"}


@router.delete("/sessions/{session_id}", summary="Close and destroy a session")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Close the session, write back high-confidence data, and remove all ephemeral data.

    Args:
        session_id: The session identifier to close.

    Returns:
        Dict with status and session_id.
    """
    sgm = _get_sgm()
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    # Merge entire session graph before closing
    try:
        await asyncio.to_thread(merge_session_to_neo4j, session_id, sgm)
    except Exception as exc:
        log.warning("Session merge failed for session %s: %s", session_id, exc)

    # Legacy write-back high-confidence data before closing
    try:
        sgm.write_back_to_neo4j(session_id, confidence_threshold=0.7)
    except Exception as exc:
        log.warning("Write-back failed for session %s: %s", session_id, exc)

    await asyncio.to_thread(sgm.close_session, session_id)
    _active_sessions.discard(session_id)
    log.info("Session deleted: %s", session_id)
    return {"status": "deleted", "session_id": session_id}


@router.post("/session/{session_id}/merge", summary="Merge session graph into persistent Neo4j")
async def merge_session(session_id: str) -> Dict[str, Any]:
    """Merge all session-scoped graph data from Memgraph into Neo4j."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    sgm = _get_sgm()
    result = await asyncio.to_thread(merge_session_to_neo4j, session_id, sgm)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/sessions/{session_id}/query", summary="Query session with SSE streaming")
async def query_session(session_id: str, req: QueryRequest) -> StreamingResponse:
    """Send a message to the LangGraph agent and stream back events via SSE."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    from agent.langgraph_agent import build_graph
    from langchain_core.messages import HumanMessage
    settings = _get_settings()

    async def _event_generator():
        graph = build_graph(
            ws_broadcaster=None,
            session_id=session_id,
            groq_api_key=settings.groq_api_key,
        )
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=req.message)]},
            config={"recursion_limit": 12},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"event: thought_token\ndata: {json.dumps({'token': token})}\n\n"
            elif kind == "on_tool_start":
                yield f"event: tool_start\ndata: {json.dumps({'tool': event['name'], 'input': str(event['data'].get('input', ''))[:200]})}\n\n"
            elif kind == "on_tool_end":
                result = str(event["data"].get("output", ""))
                yield f"event: tool_result\ndata: {json.dumps({'tool': event['name'], 'summary': result[:500]})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.get(
    "/sessions/{session_id}/graph",
    response_model=GraphResponse,
    summary="Get the current session subgraph",
)
async def get_session_graph(session_id: str) -> GraphResponse:
    """Return the D3.js-ready subgraph for the given session.

    Args:
        session_id: The session identifier.

    Returns:
        GraphResponse with nodes and edges.
    """
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    sgm = _get_sgm()
    subgraph: dict = await asyncio.to_thread(sgm.get_session_graph, session_id)
    return GraphResponse(
        nodes=subgraph.get("nodes", []),
        edges=subgraph.get("edges", []),
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint — streaming reasoning steps
# ---------------------------------------------------------------------------


@router.websocket("/sessions/{session_id}/stream")
async def ws_stream(websocket: WebSocket, session_id: str):
    """Stream LangGraph agent reasoning over WebSocket."""
    await websocket.accept()
    _ws_connections[session_id] = websocket
    try:
        data = await websocket.receive_json()
        message = data.get("message", "")
        if not message:
            await websocket.send_json({"type": "error", "error": "empty message"})
            return

        sgm = _get_sgm()
        settings = _get_settings()

        # Create session if new
        is_new = session_id not in _active_sessions
        if is_new:
            log.info("WS: Creating new session: %s", session_id)
            await asyncio.to_thread(sgm.create_session, session_id)
            _active_sessions.add(session_id)

        await websocket.send_json({"type": "status", "status": "thinking"})

        from agent.langgraph_agent import build_graph
        from langchain_core.messages import HumanMessage

        async def broadcaster(sid: str, payload: dict) -> None:
            try:
                await websocket.send_json(payload)
            except Exception:
                pass

        graph = build_graph(
            ws_broadcaster=broadcaster,
            session_id=session_id,
            groq_api_key=settings.groq_api_key,
        )

        full_answer = ""
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": 12},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    await websocket.send_json({"type": "thought_token", "token": token})
                    full_answer += token

            elif kind == "on_tool_start":
                await websocket.send_json({
                    "type": "tool_start",
                    "tool": event["name"],
                    "input": str(event["data"].get("input", ""))[:200],
                })

            elif kind == "on_tool_end":
                result = str(event["data"].get("output", ""))
                await websocket.send_json({
                    "type": "tool_result",
                    "tool": event["name"],
                    "summary": result[:500],
                })

        await websocket.send_json({
            "type": "answer",
            "answer": full_answer,
            "session_id": session_id,
        })

    except WebSocketDisconnect:
        log.info("WS: Client disconnected for session %s", session_id)
    except Exception as exc:
        log.error("WS: Error for session %s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        _ws_connections.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# Keep old WS endpoint for backward compatibility
@router.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """Stream reasoning steps over WebSocket (legacy endpoint for backward compatibility)."""
    await ws_stream(websocket, session_id)
