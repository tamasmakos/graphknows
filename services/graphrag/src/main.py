"""
HTTP interface for the KG Agent.

Exposes retrieval logic as an API, backed by the graph-based workflow.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infrastructure.config import get_app_config
from src.infrastructure.graph_db import get_database_client, GraphDB
from src.workflow.graph_workflow import GraphWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("KG Agent initialized.")

# ── Langfuse / OpenTelemetry tracing ─────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    _cfg = get_app_config()
    if _cfg.langfuse_public_key and _cfg.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = _cfg.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = _cfg.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = _cfg.langfuse_host

        _auth = base64.b64encode(
            f"{_cfg.langfuse_public_key}:{_cfg.langfuse_secret_key}".encode()
        ).decode()
        _endpoint = f"{_cfg.langfuse_host}/api/public/otel/v1/traces"

        _exporter = OTLPSpanExporter(
            endpoint=_endpoint,
            headers={"Authorization": f"Basic {_auth}"},
        )
        _provider = TracerProvider(
            resource=Resource(attributes={"service.name": "graphrag-service"})
        )
        _provider.add_span_processor(BatchSpanProcessor(_exporter))
        trace.set_tracer_provider(_provider)
        LlamaIndexInstrumentor().instrument()
        logger.info("Langfuse OTEL instrumentation enabled → %s", _endpoint)
    else:
        logger.warning("Langfuse keys not set — tracing disabled.")
except ImportError:
    logger.warning("openinference-instrumentation-llama-index not installed — tracing disabled.")
except Exception as exc:
    logger.error("Langfuse init failed: %s", exc)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="KG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    messages: List[Message] = []
    database: str = "neo4j"
    create_plot: bool = False
    accumulated_graph_data: Optional[Dict[str, Any]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_single_column(
    results: List[Dict[str, Any]], preferred_key: str
) -> List[str]:
    values: List[str] = []
    for row in results:
        val = row.get(preferred_key) or next(iter(row.values()), None)
        if isinstance(val, str):
            values.append(val)
    return values


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/schema")
async def get_schema(database: str = "neo4j"):
    """Inspect the live graph schema."""
    config = get_app_config()
    try:
        db: GraphDB = get_database_client(config, database)
        try:
            node_labels = _extract_single_column(
                db.query("CALL db.labels()"), "label"
            )
            rel_types = _extract_single_column(
                db.query("CALL db.relationshipTypes()"), "relationshipType"
            )
            prop_keys = _extract_single_column(
                db.query("CALL db.propertyKeys()"), "propertyKey"
            )
            return {
                "database": database,
                "node_labels": sorted(set(node_labels)),
                "relationship_types": sorted(set(rel_types)),
                "property_keys": sorted(set(prop_keys)),
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("Error fetching schema: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {exc}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Unified chat endpoint using the graph workflow."""
    start_time = time.time()
    try:
        from llama_index.core.llms import ChatMessage, MessageRole

        workflow = GraphWorkflow(timeout=60, verbose=True)

        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        li_messages = [
            ChatMessage(
                role=role_map.get(msg.role.lower(), MessageRole.USER),
                content=msg.content,
            )
            for msg in request.messages
        ]

        result = await workflow.run(query=request.query, messages=li_messages)
        execution_time = time.time() - start_time

        return {
            "answer": result.get("answer"),
            "context": result.get("context"),
            "full_prompt": "",
            "execution_time": execution_time,
            "graph_data": result.get("graph_data", {"nodes": [], "edges": []}),
            "graph_stats": {},
            "query_memory_mb": {},
            "reasoning_chain": result.get("trace", []),
            "seed_entities": result.get("seed_entities", []),
            "seed_topics": result.get("seed_topics", []),
            "step_timings": result.get("step_timings", {}),
            "cypher_query": "Workflow Execution",
            "confidence_score": 1.0,
        }
    except Exception as exc:
        logger.error("Endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/chat")
async def agent_chat_endpoint(request: ChatRequest):
    """Alias for /chat."""
    return await chat_endpoint(request)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/node-connections/{node_id}")
async def get_node_connections(node_id: str, database: str = "neo4j"):
    """Fetch all immediate neighbours of a given node."""
    config = get_app_config()
    try:
        db: GraphDB = get_database_client(config, database)
        try:
            cypher = """
            MATCH (n)-[r]-(m)
            WHERE toString(id(n)) = $node_id
            RETURN n, r, m
            """
            results = db.query(cypher, {"node_id": node_id})

            nodes_dict: Dict[str, Any] = {}
            edges_list: List[Dict[str, Any]] = []

            def _node_data(node: Any) -> Dict[str, Any]:
                if hasattr(node, "properties"):
                    data = dict(node.properties)
                    data["labels"] = list(getattr(node, "labels", []))
                    data["element_id"] = str(getattr(node, "id", getattr(node, "element_id", "")))
                else:
                    data = dict(node) if node else {}
                    data.setdefault("labels", [])
                    data.setdefault("element_id", "")
                return data

            def _edge_data(edge: Any) -> Dict[str, Any]:
                if hasattr(edge, "properties"):
                    data = dict(edge.properties)
                    data["type"] = getattr(edge, "type", "")
                    data["start"] = str(getattr(edge, "src_node", ""))
                    data["end"] = str(getattr(edge, "dest_node", ""))
                else:
                    data = dict(edge) if edge else {}
                    data.setdefault("type", "")
                    data.setdefault("start", "")
                    data.setdefault("end", "")
                return data

            for row in results:
                for key in ("n", "m"):
                    nd = _node_data(row.get(key))
                    if nd.get("element_id"):
                        nodes_dict[nd["element_id"]] = nd
                ed = _edge_data(row.get("r"))
                if ed.get("start") and ed.get("end"):
                    edges_list.append(ed)

            return {"nodes": list(nodes_dict.values()), "edges": edges_list}
        finally:
            db.close()
    except Exception as exc:
        logger.error("Error fetching node connections: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
