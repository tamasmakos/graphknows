"""
HTTP interface for the GraphRAG Agent Service.

Endpoints:
  POST /chat         — streaming SSE chat (fetch + ReadableStream)
  POST /chat/sync    — non-streaming JSON chat (for testing)
  GET  /schema       — live graph schema inspection
  GET  /health
"""
from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ── Langfuse / OpenTelemetry ──────────────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    _cfg = __import__("src.common.config.settings", fromlist=["AppSettings"]).AppSettings()
    if _cfg.langfuse_public_key and _cfg.langfuse_secret_key:
        os.environ.update({
            "LANGFUSE_PUBLIC_KEY": _cfg.langfuse_public_key,
            "LANGFUSE_SECRET_KEY": _cfg.langfuse_secret_key,
            "LANGFUSE_HOST": _cfg.langfuse_host,
        })
        _auth = base64.b64encode(
            f"{_cfg.langfuse_public_key}:{_cfg.langfuse_secret_key}".encode()
        ).decode()
        _endpoint = f"{_cfg.langfuse_host}/api/public/otel/v1/traces"
        _exporter = OTLPSpanExporter(
            endpoint=_endpoint,
            headers={"Authorization": f"Basic {_auth}"},
        )
        _provider = TracerProvider(resource=Resource(attributes={"service.name": "graphrag"}))
        _provider.add_span_processor(BatchSpanProcessor(_exporter))
        trace.set_tracer_provider(_provider)
        LlamaIndexInstrumentor().instrument()
        logger.info("Langfuse OTEL tracing enabled → %s", _endpoint)
    else:
        logger.warning("Langfuse keys not set — tracing disabled.")
except Exception as exc:
    logger.warning("Tracing setup skipped: %s", exc)

# ── Lifespan ──────────────────────────────────────────────────────────────────
_driver = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver
    from src.infrastructure.neo4j_driver import create_driver
    _driver = create_driver()
    try:
        await _driver.verify_connectivity()
        logger.info("Neo4j driver connected.")
    except Exception as exc:
        logger.warning("Neo4j not reachable at startup: %s", exc)
    yield
    if _driver:
        await _driver.close()
        logger.info("Neo4j driver closed.")


app = FastAPI(title="GraphRAG Agent Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    messages: List[Message] = []
    conversation_id: Optional[str] = None
    database: str = "neo4j"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_streaming(request: ChatRequest):
    """
    Streaming SSE endpoint.
    Client reads via fetch() + ReadableStream (POST-based, EventSource not used).
    """
    if _driver is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    from src.agent.workflow import stream_agent
    from llama_index.core.llms import ChatMessage, MessageRole

    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM}
    history = [
        ChatMessage(role=role_map.get(m.role.lower(), MessageRole.USER), content=m.content)
        for m in request.messages
    ]

    async def event_stream() -> AsyncIterator[bytes]:
        async for chunk in stream_agent(request.query, _driver, request.database, history):
            yield f"data: {chunk}\n\n".encode()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync(request: ChatRequest):
    """
    Non-streaming JSON endpoint — returns a complete AgentResponse.
    Useful for testing and for clients that don't support SSE.
    """
    if _driver is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    from src.agent.workflow import run_agent
    from llama_index.core.llms import ChatMessage, MessageRole

    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM}
    history = [
        ChatMessage(role=role_map.get(m.role.lower(), MessageRole.USER), content=m.content)
        for m in request.messages
    ]

    try:
        result = await run_agent(request.query, _driver, request.database, history)
        return result
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/schema")
async def get_schema(database: str = "neo4j"):
    """Inspect the live graph schema."""
    if _driver is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        async with _driver.session(database=database) as session:
            labels_r = await session.run("CALL db.labels()")
            rels_r = await session.run("CALL db.relationshipTypes()")
            props_r = await session.run("CALL db.propertyKeys()")

            labels = [r["label"] async for r in labels_r]
            rels = [r["relationshipType"] async for r in rels_r]
            props = [r["propertyKey"] async for r in props_r]

        return {
            "database": database,
            "node_labels": sorted(set(labels)),
            "relationship_types": sorted(set(rels)),
            "property_keys": sorted(set(props)),
        }
    except Exception as exc:
        logger.error("Schema fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/node-connections/{node_id}")
async def get_node_connections(node_id: str, database: str = "neo4j"):
    """Fetch all immediate neighbours of a node."""
    if _driver is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    cypher = """
    MATCH (n)-[r]-(m)
    WHERE toString(id(n)) = $node_id
    RETURN n, r, m
    LIMIT 100
    """
    try:
        async with _driver.session(database=database) as session:
            result = await session.run(cypher, {"node_id": node_id})
            records = await result.data()

        nodes_dict: Dict[str, Any] = {}
        edges: List[Dict[str, Any]] = []

        for row in records:
            for key in ("n", "m"):
                node = row.get(key)
                if node:
                    nid = str(node.element_id if hasattr(node, "element_id") else node.id)
                    nodes_dict[nid] = {
                        "element_id": nid,
                        "labels": list(node.labels) if hasattr(node, "labels") else [],
                        **dict(node),
                    }
            rel = row.get("r")
            if rel:
                edges.append({
                    "type": rel.type if hasattr(rel, "type") else "",
                    "start": str(rel.start_node.element_id if hasattr(rel, "start_node") else ""),
                    "end": str(rel.end_node.element_id if hasattr(rel, "end_node") else ""),
                    **dict(rel),
                })

        return {"nodes": list(nodes_dict.values()), "edges": edges}
    except Exception as exc:
        logger.error("Node connections error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
