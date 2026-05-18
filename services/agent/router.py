"""Router — all HTTP + WS route handlers for the agent service."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from starlette.websockets import WebSocketState

from agent.api.schemas import ChatRequest, ChatResponse, GraphResponse, QueryRequest  # noqa: F401
from agent.config import AgentSettings
try:
    try:
        from agent.graph.events import (
            ClaimVerifiedEvent,
            ContradictionDetectedEvent,
            EvidenceAddedEvent,
            GraphBuildingEvent,
            GraphReadyEvent,
            PhaseChangeEvent,
            ReportReadyEvent,
            ThreadStartedEvent,
        )
        from agent.graph.state import InvestigativeState, build_initial_investigative_state
        from agent.graph.workflow import REQUIRED_STREAM_EVENTS
    except ImportError:
        from services.agent.graph.events import (
            ClaimVerifiedEvent,
            ContradictionDetectedEvent,
            EvidenceAddedEvent,
            GraphBuildingEvent,
            GraphReadyEvent,
            PhaseChangeEvent,
            ReportReadyEvent,
            ThreadStartedEvent,
        )
        from services.agent.graph.state import InvestigativeState, build_initial_investigative_state
        from services.agent.graph.workflow import REQUIRED_STREAM_EVENTS
except Exception:
    ClaimVerifiedEvent = None  # type: ignore
    ContradictionDetectedEvent = None  # type: ignore
    EvidenceAddedEvent = None  # type: ignore
    GraphBuildingEvent = None  # type: ignore
    GraphReadyEvent = None  # type: ignore
    PhaseChangeEvent = None  # type: ignore
    ReportReadyEvent = None  # type: ignore
    ThreadStartedEvent = None  # type: ignore
    InvestigativeState = None  # type: ignore
    build_initial_investigative_state = None  # type: ignore
    REQUIRED_STREAM_EVENTS = ()  # type: ignore
from agent.session.manager import SessionGraphManager
from agent.session.end import merge_session_to_neo4j
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
_search_client: Any | None = None
_graph_backend = None
_tavily: Any | None = None
_cellar: Any | None = None
_extractor: Any | None = None
_ws_connections: dict[str, WebSocket] = {}


def _websocket_is_open(ws: WebSocket) -> bool:
    return (
        ws.client_state != WebSocketState.DISCONNECTED
        and ws.application_state != WebSocketState.DISCONNECTED
    )


async def _safe_ws_send_json(ws: WebSocket, payload: dict[str, Any]) -> bool:
    if not _websocket_is_open(ws):
        return False
    try:
        await ws.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if 'close message has been sent' in str(exc):
            return False
        raise


async def _safe_ws_send_text(ws: WebSocket, payload: str) -> bool:
    if not _websocket_is_open(ws):
        return False
    try:
        await ws.send_text(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if 'close message has been sent' in str(exc):
            return False
        raise


async def _broadcast_graph_update(session_id: str, payload: dict) -> None:
    ws = _ws_connections.get(session_id)
    if ws:
        try:
            sent = await _safe_ws_send_text(ws, json.dumps({"type": "graph_update", **payload}))
            if not sent:
                _ws_connections.pop(session_id, None)
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


def _get_search_client() -> Any | None:
    return _search_client


def _get_graph_backend():
    return _graph_backend

def _get_extractor():
    return _extractor


# Lazy-initialised ontology singletons (created on first ws_stream call to avoid startup delay)
_ontology_router: Any | None = None
_rel_extractor: Any | None = None


def _get_ontology_components() -> tuple[Any | None, Any | None]:
    """Return (OntologyRouter, OntologyRelExtractor), initialising lazily."""
    global _ontology_router, _rel_extractor
    if _ontology_router is None:
        try:
            from graphknows.shared.ontology.loader import OntologyLoader
            from graphknows.shared.ontology.router import OntologyRouter
            from graphknows.shared.ontology.rel_extractor import OntologyRelExtractor
            _ontology_router = OntologyRouter(OntologyLoader())
            _rel_extractor = OntologyRelExtractor()
            log.info("Ontology components initialised")
        except Exception as exc:
            log.warning("Ontology components unavailable (non-fatal): %s", exc)
    return _ontology_router, _rel_extractor


def _init_router(settings: AgentSettings, sgm: SessionGraphManager, http_client: httpx.AsyncClient, search_client: Any | None = None, graph_backend=None, tavily: Any | None = None, cellar: Any | None = None, extractor: Any | None = None) -> None:
    """Initialize router singletons. Called from main.py lifespan."""
    global _settings, _sgm, _http_client, _search_client, _graph_backend, _tavily, _cellar, _extractor
    _settings = settings
    _sgm = sgm
    _http_client = http_client
    _search_client = search_client
    _graph_backend = graph_backend
    _tavily = tavily
    _cellar = cellar
    _extractor = extractor


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
# Pydantic schemas are in agent.api.schemas — re-exported here for compat
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# In-memory session registry
# ---------------------------------------------------------------------------

_active_sessions: set[str] = set()


def _normalize_session_graph_token(value: str | None, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").upper()
    return normalized or fallback


def _state_value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _build_sources_payload(raw_docs: list[Any] | None, retrieval_hits: list[Any] | None) -> list[dict[str, Any]]:
    raw_doc_index: dict[str, dict[str, Any]] = {}
    for raw_doc in raw_docs or []:
        doc_id = str(_state_value(raw_doc, "id") or _state_value(raw_doc, "doc_id") or "").strip()
        if not doc_id:
            continue
        raw_doc_index[doc_id] = {
            "id": doc_id,
            "title": str(_state_value(raw_doc, "title") or doc_id),
            "url": str(_state_value(raw_doc, "url") or ""),
            "source": str(_state_value(raw_doc, "source") or ""),
            "date": str(_state_value(raw_doc, "date") or ""),
            "text": str(_state_value(raw_doc, "text") or "")[:300],
        }

    sources: list[dict[str, Any]] = []
    ordered_hits = list(retrieval_hits or [])

    if ordered_hits:
        for index, hit in enumerate(ordered_hits, 1):
            doc_id = str(_state_value(hit, "doc_id") or _state_value(hit, "id") or "").strip()
            metadata = _state_value(hit, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            raw_doc = raw_doc_index.get(doc_id, {})
            title = str(raw_doc.get("title") or metadata.get("title") or doc_id or f"Document {index}")
            url = str(raw_doc.get("url") or metadata.get("url") or "")
            source = str(raw_doc.get("source") or metadata.get("source") or "")
            date = str(raw_doc.get("date") or metadata.get("date") or "")
            snippet = str(_state_value(hit, "text") or raw_doc.get("text") or metadata.get("text") or "")[:300]
            sources.append({
                "citation_id": f"Doc-{index}",
                "id": doc_id or f"doc-{index}",
                "title": title,
                "url": url,
                "source": source,
                "date": date,
                "text": snippet,
            })
        return sources

    for index, raw_doc in enumerate(raw_docs or [], 1):
        doc_id = str(_state_value(raw_doc, "id") or _state_value(raw_doc, "doc_id") or f"doc-{index}")
        sources.append({
            "citation_id": f"Doc-{index}",
            "id": doc_id,
            "title": str(_state_value(raw_doc, "title") or doc_id),
            "url": str(_state_value(raw_doc, "url") or ""),
            "source": str(_state_value(raw_doc, "source") or ""),
            "date": str(_state_value(raw_doc, "date") or ""),
            "text": str(_state_value(raw_doc, "text") or "")[:300],
        })
    return sources


# ---------------------------------------------------------------------------
# Helper: call graphknows /retrieve (for /chat backward compatibility)
# ---------------------------------------------------------------------------


async def _call_graphrag(message: str, session_id: str) -> dict:
    """POST to graphknows service and return the parsed JSON body.

    Args:
        message: The user query to send to graphknows.
        session_id: Session identifier for context tracking.

    Returns:
        Parsed JSON response from graphknows.

    Raises:
        HTTPException: If graphknows returns an error or is unreachable.
    """
    client = _get_http_client()
    payload = {
        "query": message,
        "session_id": session_id,
        "k_entity": 10,
        "k_semantic": 20,
        "k_final": 10,
    }
    try:
        resp = await client.post(f"{_get_settings().graphknows_url}/retrieve", json=payload)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("graphknows returned HTTP %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"graphknows error {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        log.error("graphknows connection error: %s", exc)
        raise HTTPException(status_code=502, detail=f"graphknows unreachable: {exc}") from exc

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
    full hybridrag agent trace and upsert them into the Memgraph session graph.

    Args:
        session_id: The active Memgraph session identifier.
        graphrag_response: The full JSON response from the hybridrag /retrieve endpoint.
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
    """HTTP endpoint — runs the ADR-002 investigative workflow and returns a briefing."""
    from services.agent.graph.state import build_initial_investigative_state
    from services.agent.graph.workflow import build_workflow

    sgm = _get_sgm()

    session_id: str = req.session_id or str(uuid.uuid4())
    is_new = session_id not in _active_sessions

    if is_new:
        log.info("Creating new session via HTTP: %s", session_id)
        await asyncio.to_thread(sgm.create_session, session_id)
        _active_sessions.add(session_id)

    initial_state = build_initial_investigative_state(session_id, req.message)
    workflow = build_workflow(
        mg_driver=sgm.mg_driver if getattr(sgm, '_memgraph_ok', False) else None,
        search_client=_get_search_client(),
        backend=_get_graph_backend(),
        tavily=_tavily,
        cellar=_cellar,
        extractor=_get_extractor(),
    )
    result = await workflow.ainvoke(initial_state)
    sources_payload = _build_sources_payload(
        result.get("raw_docs", []),
        result.get("retrieval_hits", []),
    )

    subgraph: dict = await asyncio.to_thread(sgm.get_session_graph, session_id)

    return ChatResponse(
        answer=result.get("briefing") or "",
        session_id=session_id,
        graph=GraphResponse(
            nodes=subgraph.get("nodes", []),
            edges=subgraph.get("edges", []),
        ),
        reasoning_steps=result.get("ws_events", []),
        sources=sources_payload,
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
        settings = _get_settings()
        await asyncio.to_thread(
            merge_session_to_neo4j,
            session_id,
            sgm,
            export_html=settings.export_graph_html,
            html_output_dir=settings.graph_html_output_dir,
        )
    except Exception as exc:
        log.warning("Session merge failed for session %s: %s", session_id, exc)

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
    """Run the ADR-002 investigative workflow and stream the briefing as a single SSE event."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    from services.agent.graph.state import build_initial_investigative_state
    from services.agent.graph.workflow import build_workflow

    sgm = _get_sgm()

    async def _event_generator():
        initial_state = build_initial_investigative_state(session_id, req.message)
        workflow = build_workflow(mg_driver=sgm.mg_driver if getattr(sgm, '_memgraph_ok', False) else None, search_client=_get_search_client(), tavily=_tavily, cellar=_cellar, extractor=_get_extractor())
        result = await workflow.ainvoke(initial_state)
        briefing = result.get("briefing") or ""
        yield f"event: briefing\ndata: {json.dumps({'briefing': briefing})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/investigate", summary="Run investigative workflow with typed SSE events")
async def investigate_session(session_id: str, req: QueryRequest) -> StreamingResponse:
    """Run the evidence-led investigative workflow and stream typed events via SSE.

    Emits: phase_change, thread_started, evidence_added, graph_building,
           graph_ready, claim_verified, contradiction_detected, report_ready.

    The existing /chat and /query paths are unaffected.

    Args:
        session_id: The session identifier.
        req: Query request with the investigation question.

    Returns:
        StreamingResponse with SSE typed events.
    """
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    from agent.graph.agent import build_graph
    from langchain_core.messages import HumanMessage

    settings = _get_settings()
    sgm = _get_sgm()

    inv_state: InvestigativeState = build_initial_investigative_state(
        session_id=session_id,
        topic=req.message,
    )

    async def persist_graph(sid: str, nodes: list[dict], edges: list[dict]) -> None:
        def _persist() -> None:
            for node in nodes:
                props = dict(node.get("properties") or {})
                node_name = props.get("name") or node.get("id", "")
                sgm.add_node(
                    sid,
                    node_name,
                    _normalize_session_graph_token(node.get("label") or node.get("type"), "ENTITY"),
                    {"name": node_name},
                )
            for edge in edges:
                sgm.add_edge(
                    sid,
                    str(edge.get("source", "")),
                    str(edge.get("target", "")),
                    _normalize_session_graph_token(edge.get("type"), "RELATED_TO"),
                    dict(edge.get("properties") or {}),
                )

        await asyncio.to_thread(_persist)

    async def _event_generator():
        nonlocal inv_state

        yield _sse(
            PhaseChangeEvent(
                session_id=session_id,
                phase=inv_state["current_macro_phase"],
                node="scope_frame",
            )
        )

        graph = build_graph(
            ws_broadcaster=None,
            session_id=session_id,
            groq_api_key=settings.groq_api_key,
            persist_graph=persist_graph,
        )

        prev_phase = inv_state["current_macro_phase"]
        evidence_count = 0
        claims_seen: set[str] = set()
        contradictions_seen: set[str] = set()
        graph_building_emitted = False
        graph_ready_emitted = False
        full_answer = ""

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=req.message)]},
            config={"recursion_limit": 12},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    full_answer += token

            elif kind == "on_tool_start":
                tool_name = event["name"]

                if tool_name in (
                    "search_entities",
                    "search_chunks",
                    "get_neighbours",
                    "get_community",
                    "fuse_and_rank",
                    "get_chunks_for_entity",
                ):
                    if prev_phase == "scope":
                        prev_phase = "investigate"
                        inv_state["current_macro_phase"] = "investigate"
                        yield _sse(
                            PhaseChangeEvent(
                                session_id=session_id,
                                phase="investigate",
                                node="investigate_dispatch_threads",
                            )
                        )
                        thread_id = f"thread_{session_id[:8]}_0"
                        yield _sse(
                            ThreadStartedEvent(
                                session_id=session_id,
                                thread_id=thread_id,
                                title=f"Evidence thread for: {req.message[:60]}",
                                sub_question_id="sq_0",
                            )
                        )

                    if not graph_building_emitted and tool_name in ("search_entities", "get_neighbours"):
                        graph_building_emitted = True
                        yield _sse(
                            GraphBuildingEvent(
                                session_id=session_id,
                                namespace=session_id,
                            )
                        )

            elif kind == "on_tool_end":
                tool_name = event["name"]
                output = event["data"].get("output")

                if tool_name in ("search_chunks", "fuse_and_rank", "get_chunks_for_entity"):
                    items = output if isinstance(output, list) else []
                    for item in items[:3]:
                        if isinstance(item, dict):
                            evidence_count += 1
                            ev_id = f"ev_{session_id[:8]}_{evidence_count}"
                            doc_url = item.get("url") or item.get("doc_id", "")
                            yield _sse(
                                EvidenceAddedEvent(
                                    session_id=session_id,
                                    evidence_id=ev_id,
                                    thread_id=f"thread_{session_id[:8]}_0",
                                    summary=str(item.get("text", ""))[:200],
                                    source_authority=3,
                                    document_url=str(doc_url) if doc_url else None,
                                )
                            )

                elif tool_name in ("search_entities", "get_neighbours") and not graph_ready_emitted:
                    items = output if isinstance(output, list) else []
                    if items:
                        graph_ready_emitted = True
                        yield _sse(
                            GraphReadyEvent(
                                session_id=session_id,
                                namespace=session_id,
                                node_count=len(items),
                                edge_count=max(0, len(items) - 1),
                            )
                        )

        if inv_state["current_macro_phase"] != "communicate":
            inv_state["current_macro_phase"] = "communicate"
            yield _sse(
                PhaseChangeEvent(
                    session_id=session_id,
                    phase="communicate",
                    node="communicate_report",
                )
            )

        # ── Run communicate_node with real Tavily docs from the ReAct agent ──
        # Import here to avoid circular imports at module level
        try:
            from agent.graph.agent import _session_docs
            from services.agent.graph.state import Chunk
            from services.agent.graph.nodes.communicate import communicate_node
            import uuid as _uuid

            raw_docs = _session_docs.get(session_id, [])
            if raw_docs:
                chunks_for_brief = [
                    Chunk(
                        id=f"chunk-{_uuid.uuid4().hex[:8]}",
                        doc_id=d.get("id", d.get("url", f"doc-{i}"))[:80],
                        text=d.get("text", "") or d.get("content", ""),
                        embedding=None,
                        metadata={"source": d.get("source", "web"), "url": d.get("url", ""), "title": d.get("title", "")},
                    )
                    for i, d in enumerate(raw_docs)
                ]
                brief_state = dict(inv_state)
                brief_state["chunks"] = chunks_for_brief
                brief_state["topic"] = req.message
                comm_result = await communicate_node(brief_state, retriever=None)
                full_answer = comm_result.get("briefing", full_answer) or full_answer
        except Exception as _comm_exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("communicate_node post-agent call failed: %s", _comm_exc)

        if full_answer:
            claim_id = f"claim_{session_id[:8]}_0"
            if claim_id not in claims_seen:
                claims_seen.add(claim_id)
                yield _sse(
                    ClaimVerifiedEvent(
                        session_id=session_id,
                        claim_id=claim_id,
                        text=full_answer,
                        status="supported",
                        importance=5,
                    )
                )

        if contradictions_seen:
            first_id = next(iter(contradictions_seen))
            yield _sse(
                ContradictionDetectedEvent(
                    session_id=session_id,
                    contradiction_id=first_id,
                    claim_id=f"claim_{session_id[:8]}_0",
                    summary="Potential contradiction detected.",
                    resolution_status="open",
                )
            )

        yield _sse(
            ReportReadyEvent(
                session_id=session_id,
                findings_count=max(1, evidence_count),
                report_excerpt=full_answer,
            )
        )

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _sse(model: BaseModel) -> str:
    """Serialise a Pydantic event model as an SSE frame."""
    return f"event: {model.event}\ndata: {model.model_dump_json()}\n\n"


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
    disconnected = False
    try:
        data = await websocket.receive_json()
        query = data.get("query") or data.get("message", "")
        if not query:
            await _safe_ws_send_json(websocket, {"type": "error", "message": "empty query"})
            return

        sgm = _get_sgm()

        # Create session if new (gracefully handle Memgraph-down)
        is_new = session_id not in _active_sessions
        if is_new:
            log.info("WS: Creating new session: %s", session_id)
            try:
                await asyncio.to_thread(sgm.create_session, session_id)
            except Exception as sg_err:
                log.warning("WS: create_session failed (non-fatal, continuing): %s", sg_err)
            _active_sessions.add(session_id)

        if not await _safe_ws_send_json(websocket, {"type": "status", "status": "thinking"}):
            return

        from agent.graph.state import build_initial_investigative_state
        from agent.graph.workflow import build_workflow

        initial_state = build_initial_investigative_state(session_id, topic=query)
        ont_router, ont_extractor = _get_ontology_components()
        workflow = build_workflow(
            mg_driver=sgm.mg_driver if getattr(sgm, '_memgraph_ok', False) else None,
            search_client=_get_search_client(),
            ontology_router=ont_router,
            rel_extractor=ont_extractor,
            tavily=_tavily,
            cellar=_cellar,
            extractor=_get_extractor(),
        )
        seen: set[tuple] = set()
        max_iter = 0
        best_briefing = ""
        max_entities = 0
        max_relations = 0
        last_dup = 0.0
        latest_raw_docs: list[Any] = []
        latest_retrieval_hits: list[Any] = []
        async for chunk in workflow.astream(initial_state):
            # LangGraph astream yields {"node_name": state_update} per node
            # ws_events live inside the node's state update, not at the top level
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict):
                    continue
                max_iter = max(max_iter, node_state.get("iteration_count", 0))
                best_briefing = node_state.get("briefing") or best_briefing
                max_entities = max(max_entities, len(node_state.get("entities", [])))
                max_relations = max(max_relations, len(node_state.get("relations", [])))
                last_dup = node_state.get("dup_ratio", 0.0) or last_dup
                if node_state.get("raw_docs"):
                    latest_raw_docs = list(node_state.get("raw_docs", []))
                if node_state.get("retrieval_hits"):
                    latest_retrieval_hits = list(node_state.get("retrieval_hits", []))
                for event in node_state.get("ws_events", []):
                    key = (event.get("event_type"), event.get("ts"))
                    if key in seen:
                        continue
                    seen.add(key)
                    normalized = {
                        "type": "ws_event",
                        "payload": event,
                    }
                    if not await _safe_ws_send_json(websocket, normalized):
                        disconnected = True
                        break
                if disconnected:
                    break
            if disconnected:
                break
        if disconnected:
            return
        sources_payload = _build_sources_payload(latest_raw_docs, latest_retrieval_hits)
        await _safe_ws_send_json(websocket, {
            "type": "done",
            "briefing": best_briefing,
            "iteration_count": max_iter,
            "entity_count": max_entities,
            "relation_count": max_relations,
            "dup_ratio": last_dup,
            "sources": sources_payload,
        })

    except WebSocketDisconnect:
        log.info("WS: Client disconnected for session %s", session_id)
    except Exception as exc:
        log.error("WS: Error for session %s: %s", session_id, exc)
        try:
            await _safe_ws_send_json(websocket, {"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        _ws_connections.pop(session_id, None)
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass


# Keep old WS endpoint for backward compatibility
@router.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """Stream reasoning steps over WebSocket (legacy endpoint for backward compatibility)."""
    await ws_stream(websocket, session_id)
