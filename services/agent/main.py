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
import hashlib
import logging
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    "PGVECTOR_DSN",
    "HTTPX_TIMEOUT",
}


def _load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "graphrag_url": os.environ.get("GRAPHRAG_URL", "http://localhost:8001"),
        "memgraph_uri": os.environ.get("MEMGRAPH_URI", "bolt://172.17.0.1:17687"),
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        "pgvector_dsn": os.environ.get(
            "PGVECTOR_DSN",
            "postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb",
        ),
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

_sgm = None  # SessionGraphManager instance
_http_client = None  # shared httpx.AsyncClient

# In-memory session registry — maps session_id → True (session is active)
_active_sessions: set[str] = set()
_session_graph_views: dict[str, dict[str, Any]] = {}


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
        pgvector_dsn=CONFIG["pgvector_dsn"],
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


class GraphViewsResponse(BaseModel):
    knowledge: GraphResponse
    lexical: GraphResponse
    active: str = "knowledge"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    graph: GraphResponse
    graph_views: Optional[GraphViewsResponse] = None
    reasoning_steps: list = []
    sources: list = []


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
    """Map tool name to a frontend-friendly observation type label."""
    mapping = {
        "search_entities": "entities",
        "search_chunks": "chunks",
        "get_community": "community",
        "get_neighbours": "neighbours",
        "fuse_and_rank": "ranked",
        "get_chunks_for_entity": "chunks",
        "get_triplets": "triplets",
        "get_source_document": "source_document",
    }
    return mapping.get(tool_name, "unknown")


def _normalise_chunk_id(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return text.replace("chunk-", "", 1).rsplit("-", 1)[0] if text.startswith("chunk-") else text


def _make_edge_id(parts: list[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    safe = "-".join(p.replace(" ", "_") for p in parts[:3])
    return f"edge-{safe}-{digest}"


def _build_graph_views(
    subgraph: dict[str, list], reasoning_steps: list[dict], sources: list[dict]
) -> dict[str, dict[str, list]]:
    nodes = subgraph.get("nodes", []) or []
    edges = subgraph.get("edges", []) or []
    node_by_id = {node.get("id"): node for node in nodes if node.get("id")}

    lexical_chunk_ids = {
        str(src.get("id") or src.get("chunk_id") or "")
        for src in sources
        if str(src.get("id") or src.get("chunk_id") or "")
    }
    for step in reasoning_steps:
        tool_name = step.get("tool_name") or _extract_tool_name(step.get("action"))
        obs = step.get("observation")
        if tool_name in {"search_chunks", "fuse_and_rank", "get_chunks_for_entity"} and isinstance(
            obs, list
        ):
            for item in obs:
                if isinstance(item, dict):
                    cid = str(item.get("chunk_id") or item.get("id") or "")
                    if cid:
                        lexical_chunk_ids.add(cid)

    if not lexical_chunk_ids:
        lexical_chunk_ids = {
            _normalise_chunk_id(
                str((node.get("properties") or {}).get("id") or node.get("id") or "")
            )
            for node in nodes
            if (node.get("label") == "Chunk" or node.get("type") == "Chunk")
        }
        lexical_chunk_ids.discard("")

    lexical_nodes_map: dict[str, dict] = {}
    lexical_edges_map: dict[str, dict] = {}

    def add_lexical_node(node_id: str) -> None:
        node = node_by_id.get(node_id)
        if node and node_id not in lexical_nodes_map:
            lexical_nodes_map[node_id] = node

    for edge in edges:
        edge_type = edge.get("type")
        source_id = (
            edge.get("source", {}).get("id")
            if isinstance(edge.get("source"), dict)
            else edge.get("source")
        )
        target_id = (
            edge.get("target", {}).get("id")
            if isinstance(edge.get("target"), dict)
            else edge.get("target")
        )
        if edge_type != "MENTIONS":
            continue
        source_node = node_by_id.get(str(source_id)) if source_id else None
        chunk_id = str(
            ((source_node or {}).get("properties") or {}).get("id")
            or _normalise_chunk_id(source_id)
        )
        if chunk_id and chunk_id in lexical_chunk_ids:
            lexical_edges_map[
                str(
                    edge.get("id")
                    or _make_edge_id([str(source_id), edge_type or "MENTIONS", str(target_id)])
                )
            ] = edge
            if source_id:
                add_lexical_node(str(source_id))
            if target_id:
                add_lexical_node(str(target_id))

    relation_chunks_by_pair: dict[tuple[str, str, str], set[str]] = {}
    for step in reasoning_steps:
        tool_name = step.get("tool_name") or _extract_tool_name(step.get("action"))
        obs = step.get("observation")
        if tool_name != "get_triplets" or not isinstance(obs, list):
            continue
        for item in obs:
            if not isinstance(item, dict):
                continue
            entity_a = str(item.get("entity_a") or "").strip()
            entity_b = str(item.get("entity_b") or "").strip()
            via_chunks = item.get("via_chunks") or []
            if entity_a and entity_b:
                relation_type = str(
                    item.get("relation") or item.get("predicate") or "CO_OCCURS"
                ).strip()
                key = tuple(sorted((entity_a, entity_b)) + [relation_type])
                relation_chunks_by_pair.setdefault(key, set()).update(
                    str(cid) for cid in via_chunks if str(cid)
                )

    knowledge_nodes_map: dict[str, dict] = {}
    knowledge_edges_map: dict[str, dict] = {}

    def add_knowledge_node(node_id: str) -> None:
        node = node_by_id.get(node_id)
        if node and node_id not in knowledge_nodes_map:
            knowledge_nodes_map[node_id] = node

    for edge in edges:
        edge_type = edge.get("type")
        if edge_type not in {"CO_OCCURS", "RELATES_TO"}:
            continue
        source_id = (
            edge.get("source", {}).get("id")
            if isinstance(edge.get("source"), dict)
            else edge.get("source")
        )
        target_id = (
            edge.get("target", {}).get("id")
            if isinstance(edge.get("target"), dict)
            else edge.get("target")
        )
        if not source_id or not target_id:
            continue
        src_node = node_by_id.get(str(source_id))
        tgt_node = node_by_id.get(str(target_id))
        if not src_node or not tgt_node:
            continue
        src_name = str(
            (src_node.get("properties") or {}).get("name") or src_node.get("name") or ""
        ).strip()
        tgt_name = str(
            (tgt_node.get("properties") or {}).get("name") or tgt_node.get("name") or ""
        ).strip()
        key = (
            tuple(sorted((src_name, tgt_name)) + [edge_type])
            if src_name and tgt_name and edge_type == "CO_OCCURS"
            else None
        )
        existing_chunk_ids = {
            str(cid) for cid in ((edge.get("properties") or {}).get("chunk_ids") or []) if str(cid)
        }
        traced_chunk_ids = relation_chunks_by_pair.get(key, set()) if key else set()
        merged_chunk_ids = sorted(existing_chunk_ids | traced_chunk_ids)
        relation_label = (
            "CO_OCCURS"
            if edge_type == "CO_OCCURS"
            else str((edge.get("properties") or {}).get("predicate") or edge_type)
        )
        knowledge_edge = {
            **edge,
            "type": "KG_RELATION",
            "properties": {
                **(edge.get("properties") or {}),
                "view": "knowledge",
                "relation_type": relation_label,
                "source_name": src_name,
                "target_name": tgt_name,
                "chunk_ids": merged_chunk_ids,
                "chunk_count": len(merged_chunk_ids),
                "source_mode": "triplet_chunks"
                if traced_chunk_ids
                else (
                    "edge_provenance"
                    if existing_chunk_ids
                    else ("cooccurrence_only" if edge_type == "CO_OCCURS" else "session_relation")
                ),
            },
        }
        knowledge_edges_map[
            str(
                knowledge_edge.get("id")
                or _make_edge_id([str(source_id), relation_label, str(target_id)])
            )
        ] = knowledge_edge
        add_knowledge_node(str(source_id))
        add_knowledge_node(str(target_id))

    return {
        "knowledge": {
            "nodes": list(knowledge_nodes_map.values()),
            "edges": list(knowledge_edges_map.values()),
        },
        "lexical": {
            "nodes": list(lexical_nodes_map.values()),
            "edges": list(lexical_edges_map.values()),
        },
    }


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

        driver = _GraphDatabase.driver(
            CONFIG["neo4j_uri"],
            auth=(CONFIG["neo4j_user"], CONFIG["neo4j_password"]),
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
                names=[n.upper() for n in entity_names],
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

    Processes every trace step's observation, not just the final answer text,
    to mine the richest possible graph structure.

    Args:
        session_id: The active Memgraph session identifier.
        graphrag_response: The full JSON response from the graphrag /retrieve endpoint.
    """
    import re as _re

    entity_names: list[str] = []
    relates_to: list[tuple[str, str]] = []  # (entity_a, entity_b) co-occurrence pairs
    topic_nodes: list[dict] = []  # {id, community_id, summary, label, entity_name}
    chunk_nodes: list[dict] = []  # {chunk_id, doc_id, excerpt, date, institution, url}

    # ── Pull from explicit top-level fields ─────────────────────────────
    for name in graphrag_response.get("entities_mentioned", []):
        if isinstance(name, str):
            entity_names.append(name)
        elif isinstance(name, dict) and name.get("name"):
            entity_names.append(name["name"])

    for citation in graphrag_response.get("citations", []):
        if isinstance(citation, dict):
            # Enrich chunk node from citations
            cid = citation.get("chunk_id", "")
            doc_id = citation.get("doc_id", "")
            if cid:
                meta = _infer_doc_metadata(doc_id)
                chunk_nodes.append(
                    {
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "excerpt": citation.get("excerpt", "")[:300],
                        **meta,
                    }
                )

    # ── Parse ALL trace steps ────────────────────────────────────────────
    last_entity_list: list[str] = []  # entities seen in last search_entities step

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
            # Build co-occurrence pairs among entities in same get_neighbours call
            if tool == "get_neighbours" and last_entity_list and step_entities:
                for anchor in last_entity_list[:1]:  # use the top entity as anchor
                    for neighbour in step_entities[:8]:
                        if anchor != neighbour:
                            relates_to.append((anchor, neighbour))
            if tool == "search_entities":
                last_entity_list = step_entities
            elif tool == "get_neighbours":
                # Also pair returned neighbours with each other (shared chunk context)
                for i, ea in enumerate(step_entities[:6]):
                    for eb in step_entities[i + 1 : 6]:
                        relates_to.append((ea, eb))

        elif tool == "get_community":
            # obs is {id, community_id, summary, label} or a list thereof
            items = obs if isinstance(obs, list) else [obs]
            # Determine which entity was queried from action args
            m = _re.search(r'get_community\(["\']?([^"\'(),]+)["\']?\)', action)
            entity_name = m.group(1).strip() if m else ""
            for item in items:
                if isinstance(item, dict) and (item.get("id") or item.get("community_id")):
                    topic_nodes.append({**item, "entity_name": entity_name})

        elif tool in ("search_chunks", "fuse_and_rank", "get_chunks_for_entity") and isinstance(
            obs, list
        ):
            for item in obs:
                if isinstance(item, dict):
                    cid = item.get("chunk_id", "")
                    if cid:
                        doc_id = item.get("doc_id", "")
                        meta = _infer_doc_metadata(doc_id)
                        chunk_nodes.append(
                            {
                                "chunk_id": cid,
                                "doc_id": doc_id,
                                "excerpt": item.get("text", "")[:300],
                                **meta,
                            }
                        )

    # ── Mine [BRACKETED] names from final answer text ────────────────────
    answer_text = graphrag_response.get("answer", "")
    for match in _re.findall(r"\[([A-Z][A-Z0-9_\s]+)\]", answer_text):
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
        session_id,
        len(cooccurs_edges),
        len(unique_entities),
    )

    # De-duplicate chunk_nodes by chunk_id (last write wins)
    chunk_by_id: dict[str, dict] = {}
    for c in chunk_nodes:
        if c.get("chunk_id"):
            chunk_by_id[c["chunk_id"]] = c

    log.info(
        "Session %s enrichment: %d entities, %d relates_to, %d topics, %d chunks",
        session_id,
        len(unique_entities),
        len(relates_to),
        len(topic_nodes),
        len(chunk_by_id),
    )

    # Seed entities from Neo4j (creates Entity+Chunk nodes and MENTIONS edges)
    if unique_entities:
        _sgm.seed_from_neo4j(session_id, unique_entities)

    # Enrich with richer graph structure
    _sgm.add_graph_enrichments(
        session_id,
        relates_to=list(dict.fromkeys(relates_to)),  # de-dup preserving order
        topic_nodes=topic_nodes,
        chunk_nodes=list(chunk_by_id.values()),
    )

    # Upsert CO_OCCURS edges from Neo4j into the session graph
    if cooccurs_edges:
        _sgm.add_cooccurs_edges(session_id, cooccurs_edges)


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

    # Enrich reasoning steps with tool_name, observation_type, and truncated observation
    raw_steps: list = graphrag_resp.get("reasoning_steps", [])
    reasoning_steps: list = []
    for step in raw_steps:
        tool_name = _extract_tool_name(step.get("action"))
        obs = step.get("observation")
        obs_type = (
            _observation_type(tool_name)
            if tool_name
            else ("error" if isinstance(obs, dict) and obs.get("error") else "unknown")
        )
        # Truncate observation to avoid giant payloads
        if obs is not None:
            obs_str = (
                obs if isinstance(obs, str) else __import__("json").dumps(obs, ensure_ascii=False)
            )
            obs_truncated = obs_str[:500] + ("…" if len(obs_str) > 500 else "")
        else:
            obs_truncated = None
        reasoning_steps.append(
            {
                **{k: v for k, v in step.items() if k != "observation"},
                "tool_name": tool_name,
                "observation_type": obs_type,
                "observation": obs_truncated,
            }
        )

    # Build sources from citations / chunk references in graphrag response
    sources: list = []
    for citation in graphrag_resp.get("citations", []):
        if isinstance(citation, dict):
            src: dict = {}
            if citation.get("chunk_id"):
                src["id"] = citation["chunk_id"]
            if citation.get("excerpt"):
                src["text"] = citation["excerpt"]
            elif citation.get("text"):
                src["text"] = citation["text"]
            if citation.get("doc_id"):
                src["doc_id"] = citation["doc_id"]
            if citation.get("url"):
                src["url"] = citation["url"]
            if src.get("text"):
                sources.append(src)

    graph_views = _build_graph_views(subgraph, raw_steps, sources)
    _session_graph_views[session_id] = graph_views
    knowledge_graph = graph_views["knowledge"]
    lexical_graph = graph_views["lexical"]

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        graph=GraphResponse(
            nodes=knowledge_graph["nodes"],
            edges=knowledge_graph["edges"],
        ),
        graph_views=GraphViewsResponse(
            knowledge=GraphResponse(
                nodes=knowledge_graph["nodes"],
                edges=knowledge_graph["edges"],
            ),
            lexical=GraphResponse(
                nodes=lexical_graph["nodes"],
                edges=lexical_graph["edges"],
            ),
            active="knowledge",
        ),
        reasoning_steps=reasoning_steps,
        sources=sources,
    )


@app.get(
    "/session/{session_id}/graph",
    response_model=GraphResponse,
    summary="Get the current session subgraph",
)
async def get_session_graph(session_id: str, view: str = "knowledge") -> GraphResponse:
    """Return the current session subgraph for the requested view."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    subgraph: dict = await asyncio.to_thread(_sgm.get_session_graph, session_id)
    cached_views = _session_graph_views.get(session_id)
    if view == "raw":
        selected = subgraph
    elif cached_views and view in {"knowledge", "lexical"}:
        selected = cached_views.get(view, {"nodes": [], "edges": []})
    else:
        selected = _build_graph_views(subgraph, [], []).get(
            view if view in {"knowledge", "lexical"} else "knowledge",
            {"nodes": [], "edges": []},
        )
    return GraphResponse(
        nodes=selected.get("nodes", []),
        edges=selected.get("edges", []),
    )


@app.get(
    "/session/{session_id}/relation-sources",
    summary="Get chunk-level provenance for a graph relation",
)
async def get_relation_sources(
    session_id: str,
    relation_type: str,
    source_name: str,
    target_name: str,
    chunk_ids: str | None = None,
) -> Dict[str, Any]:
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    requested_chunk_ids = [cid for cid in (chunk_ids or "").split(",") if cid]
    cached_views = _session_graph_views.get(session_id, {})
    allowed_chunk_ids: list[str] = []
    for graph_view in cached_views.values():
        if not isinstance(graph_view, dict):
            continue
        for edge in graph_view.get("edges", []):
            if not isinstance(edge, dict):
                continue
            edge_source = (
                edge.get("source", {}).get("id")
                if isinstance(edge.get("source"), dict)
                else edge.get("source")
            )
            edge_target = (
                edge.get("target", {}).get("id")
                if isinstance(edge.get("target"), dict)
                else edge.get("target")
            )
            edge_props = edge.get("properties") or {}
            edge_relation_type = str(edge_props.get("relation_type") or edge.get("type") or "")
            edge_source_name = str(
                edge_props.get("source_name") or edge_props.get("source_entity") or edge_source
            )
            edge_target_name = str(
                edge_props.get("target_name") or edge_props.get("target_entity") or edge_target
            )
            if (
                edge_relation_type == relation_type
                and edge_source_name == source_name
                and edge_target_name == target_name
            ):
                allowed_chunk_ids = [
                    str(cid) for cid in edge_props.get("chunk_ids", []) if str(cid)
                ]
                break
        if allowed_chunk_ids:
            break
    if requested_chunk_ids:
        allowed = set(allowed_chunk_ids)
        chunk_list = [cid for cid in requested_chunk_ids if cid in allowed]
    else:
        chunk_list = allowed_chunk_ids

    sources = await asyncio.to_thread(
        _sgm.get_relation_sources,
        session_id,
        relation_type,
        source_name,
        target_name,
        chunk_list,
    )
    return {
        "session_id": session_id,
        "relation_type": relation_type,
        "source_name": source_name,
        "target_name": target_name,
        "sources": sources,
    }


@app.delete("/session/{session_id}", summary="Close and destroy a session")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Close the session and remove all ephemeral data from Memgraph."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    await asyncio.to_thread(_sgm.close_session, session_id)
    _active_sessions.discard(session_id)
    _session_graph_views.pop(session_id, None)
    log.info("Session deleted: %s", session_id)
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# WebSocket endpoint — streaming reasoning steps
# ---------------------------------------------------------------------------


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """Stream reasoning steps over WebSocket, then send final answer + graph."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        message = data.get("message", "")
        if not message:
            await websocket.send_json({"type": "error", "error": "empty message"})
            return

        # Create session if new
        is_new = session_id not in _active_sessions
        if is_new:
            log.info("WS: Creating new session: %s", session_id)
            await asyncio.to_thread(_sgm.create_session, session_id)
            _active_sessions.add(session_id)
        else:
            log.info("WS: Continuing session: %s", session_id)

        await websocket.send_json({"type": "status", "status": "thinking"})

        # Call graphrag (async)
        graphrag_resp = await _call_graphrag(message, session_id)

        # Stream reasoning steps
        for step in graphrag_resp.get("reasoning_steps", []):
            await websocket.send_json(
                {
                    "type": "step",
                    "step_num": step.get("step", 0),
                    "thought": step.get("thought", ""),
                    "action": step.get("action", ""),
                    "observation": str(step.get("observation", ""))[:200],
                }
            )
            await asyncio.sleep(0)  # yield to event loop

        # Upsert entities
        await asyncio.to_thread(_upsert_entities, session_id, graphrag_resp)

        # Get graph
        subgraph = await asyncio.to_thread(_sgm.get_session_graph, session_id)

        # Enrich reasoning steps like the HTTP path
        raw_steps = graphrag_resp.get("reasoning_steps", [])
        reasoning_steps: list[dict[str, Any]] = []
        for step in raw_steps:
            tool_name = _extract_tool_name(step.get("action"))
            obs = step.get("observation")
            obs_type = (
                _observation_type(tool_name)
                if tool_name
                else ("error" if isinstance(obs, dict) and obs.get("error") else "unknown")
            )
            reasoning_steps.append(
                {
                    **step,
                    "tool_name": tool_name,
                    "observation_type": obs_type,
                }
            )

        sources: list[dict[str, Any]] = []
        for citation in graphrag_resp.get("citations", []):
            if isinstance(citation, dict):
                src: dict[str, Any] = {}
                if citation.get("chunk_id"):
                    src["id"] = citation["chunk_id"]
                if citation.get("excerpt"):
                    src["text"] = citation["excerpt"]
                elif citation.get("text"):
                    src["text"] = citation["text"]
                if citation.get("doc_id"):
                    src["doc_id"] = citation["doc_id"]
                if citation.get("url"):
                    src["url"] = citation["url"]
                if src.get("text"):
                    sources.append(src)

        graph_views = _build_graph_views(
            subgraph, graphrag_resp.get("reasoning_steps", []), sources
        )
        _session_graph_views[session_id] = graph_views
        knowledge_graph = graph_views["knowledge"]

        await websocket.send_json(
            {
                "type": "answer",
                "answer": graphrag_resp.get("answer", ""),
                "graph": knowledge_graph,
                "graph_views": {
                    "knowledge": knowledge_graph,
                    "lexical": graph_views["lexical"],
                    "active": "knowledge",
                },
                "reasoning_steps": reasoning_steps,
                "sources": sources,
                "session_id": session_id,
            }
        )
    except WebSocketDisconnect:
        log.info("WS: Client disconnected for session %s", session_id)
    except Exception as exc:
        log.error("WS: Error for session %s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
