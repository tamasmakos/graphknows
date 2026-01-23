"""
HTTP interface for the KG Agent.

This module defines the FastAPI app for the Agent, which exposes
retrieval logic as tools and serves the frontend.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infrastructure.config import get_app_config
from src.infrastructure.graph_db import get_database_client, GraphDB
from src.workflow.graph_workflow import GraphWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("KG Agent initialized. Logging set to INFO level.")

# Initialize Langfuse Tracing
try:
    from langfuse.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()
    logger.info("Langfuse instrumentation initialized.")
except ImportError:
    logger.warning("langfuse-llama-index not found. Tracing disabled.")
except Exception as e:
    logger.error(f"Failed to initialize Langfuse instrumentation: {e}")

app = FastAPI(title="KG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    messages: List[Message] = []
    database: str = "falkordb"
    create_plot: bool = False
    accumulated_graph_data: Optional[Dict[str, Any]] = None
    use_agent: bool = False  # Deprecated, kept for compatibility

def _extract_single_column(results: List[Dict[str, Any]], preferred_key: str) -> List[str]:
    """Utility to pull a single column out of a list of row dicts."""
    values: List[str] = []
    for row in results:
        if preferred_key in row:
            val = row[preferred_key]
        else:
            val = next(iter(row.values()), None)
        if isinstance(val, str):
            values.append(val)
    return values


@app.get("/schema")
async def get_schema(database: str = "falkordb"):
    """Dynamically inspect the graph schema."""
    config = get_app_config()
    try:
        db: GraphDB = get_database_client(config, database)
        try:
            node_labels_raw = db.query("CALL db.labels()")
            rel_types_raw = db.query("CALL db.relationshipTypes()")
            prop_keys_raw = db.query("CALL db.propertyKeys()")

            node_labels = _extract_single_column(node_labels_raw, "label")
            relationship_types = _extract_single_column(rel_types_raw, "relationshipType")
            property_keys = _extract_single_column(prop_keys_raw, "propertyKey")

            return {
                "database": database,
                "node_labels": sorted(set(node_labels)),
                "relationship_types": sorted(set(relationship_types)),
                "property_keys": sorted(set(property_keys)),
            }
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        logger.error("Error fetching schema for %s: %s", database, e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {e}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint using the Event-Driven Workflow.
    """
    start_time = time.time()
    
    try:
        workflow = GraphWorkflow(timeout=60, verbose=True)
        
        # Convert pydantic messages to langchain format if needed, 
        # but the workflow step handles extraction. 
        # The input event expects raw messages or specific format.
        # GraphWorkflow.extract_keywords uses ev.get("messages")
        
        result = await workflow.run(
            query=request.query, 
            messages=request.messages
        )
        
        execution_time = time.time() - start_time
        
        # Result is from StopEvent
        # {"answer": ..., "context": ..., "graph_data": ..., "trace": ...}
        
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
            "cypher_query": "Workflow Execution",
            "confidence_score": 1.0,
        }

    except Exception as e:  # noqa: BLE001
        logger.error("Endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat")
async def agent_chat_endpoint(request: ChatRequest):
    """
    Redirects to the main chat endpoint.
    """
    return await chat_endpoint(request)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/node-connections/{node_id}")
async def get_node_connections(node_id: str, database: str = "falkordb"):
    """Fetch all immediate connections (neighbors) of a given node."""
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
            
            nodes_dict = {}
            edges_list = []
            
            def extract_node_data(node):
                if hasattr(node, "properties"):
                    data = dict(node.properties)
                    data["labels"] = list(node.labels) if hasattr(node, "labels") else []
                    data["element_id"] = str(node.id) if hasattr(node, "id") else node.element_id if hasattr(node, "element_id") else ""
                else:
                    data = dict(node) if node else {}
                    data["labels"] = data.get("labels", [])
                    data["element_id"] = data.get("element_id", "")
                return data
            
            def extract_edge_data(edge):
                if hasattr(edge, "properties"):
                    data = dict(edge.properties)
                    data["type"] = edge.type if hasattr(edge, "type") else ""
                    data["start"] = str(edge.src_node) if hasattr(edge, "src_node") else ""
                    data["end"] = str(edge.dest_node) if hasattr(edge, "dest_node") else ""
                else:
                    data = dict(edge) if edge else {}
                    data["type"] = data.get("type", "")
                    data["start"] = data.get("start", "")
                    data["end"] = data.get("end", "")
                return data
            
            for row in results:
                n = row.get("n")
                if n:
                    node_data = extract_node_data(n)
                    if node_data["element_id"]:
                        nodes_dict[node_data["element_id"]] = node_data
                m = row.get("m")
                if m:
                    node_data = extract_node_data(m)
                    if node_data["element_id"]:
                        nodes_dict[node_data["element_id"]] = node_data
                r = row.get("r")
                if r:
                    edge_data = extract_edge_data(r)
                    if edge_data["start"] and edge_data["end"]:
                        edges_list.append(edge_data)
            
            return {
                "nodes": list(nodes_dict.values()),
                "edges": edges_list
            }
        finally:
            db.close()
    except Exception as e:
        logger.error("Error fetching node connections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


static_dir = os.path.join(os.path.dirname(__file__), "frontend")
static_dir = os.path.abspath(static_dir)
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")