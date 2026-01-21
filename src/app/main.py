"""
HTTP interface for the KG Agent.

This module defines the FastAPI app for the Agent, which exposes
retrieval logic as tools and serves the frontend.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.app.infrastructure.config import get_app_config
from src.app.infrastructure.graph_db import get_database_client, GraphDB
from src.app.services.retrieval import Message, run_focused_retrieval
from src.app.agent.llamaindex_agent import get_agent
from src.app.services.retrieval import Message, run_focused_retrieval
from src.app.agent.llamaindex_agent import get_agent
from src.app.services.graph_context import init_graph_context, get_accumulated_data, get_text_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Force override any existing handlers
)
logger = logging.getLogger(__name__)
logger.info("KG Agent initialized. Logging set to INFO level.")
print("KG Agent is starting up... Visibility check.", flush=True)

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


class ChatRequest(BaseModel):
    query: str
    messages: List[Message] = []
    database: str = "falkordb"
    create_plot: bool = False
    accumulated_graph_data: Optional[Dict[str, Any]] = None
    use_agent: bool = False  # New: use LlamaIndex agent instead of direct retrieval


def _extract_single_column(results: List[Dict[str, Any]], preferred_key: str) -> List[str]:
    """
    Utility to pull a single column out of a list of row dicts.

    FalkorDB returns column names like `label`, `relationshipType`,
    and `propertyKey`, but we defensively fall back to the first value in each row.
    """
    values: List[str] = []
    for row in results:
        if preferred_key in row:
            val = row[preferred_key]
        else:
            # Fallback: take the first column value, if any.
            val = next(iter(row.values()), None)
        if isinstance(val, str):
            values.append(val)
    return values


@app.get("/schema")
async def get_schema(database: str = "falkordb"):
    """
    Dynamically inspect the graph schema for the requested database.

    This works for FalkorDB by issuing standard schema
    inspection procedures.
    """
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
    Chat endpoint supporting both direct retrieval and LlamaIndex agent modes.
    
    Set use_agent=True to use the proactive LlamaIndex agent that explores
    the graph before answering. Otherwise, uses direct retrieval pipeline.
    """
    import time
    start_time = time.time()
    
    # Use LlamaIndex agent if requested
    if request.use_agent:
        try:
            agent = get_agent(verbose=False)
            
            # Convert messages to history format
            history = []
            for msg in request.messages:
                history.append({"role": msg.role, "content": msg.content})
            
            # Initialize graph context with any accumulated data
            init_graph_context(request.accumulated_graph_data)
            
            # Run agent
            result = await agent.chat(request.query, history)
            
            # Get accumulated graph data
            graph_data = get_accumulated_data()
            
            execution_time = time.time() - start_time
            
            return {
                "answer": result["answer"],
                "context": "",  # Agent manages its own context
                "execution_time": execution_time,
                "execution_time": execution_time,
                "graph_data": graph_data,  # Return collected graph data
                "graph_stats": None,
                "query_memory_mb": None,
                "reasoning_chain": result.get("reasoning_timeline", [
                    f"Mode: LlamaIndex Agent (Tracing Disabled or Empty)",
                    f"Tools used: {len(result.get('tool_calls', []))}",
                    f"Response time: {execution_time:.2f}s",
                ] + [f"Tool: {tc['tool']}" for tc in result.get('tool_calls', [])]),
                "cypher_query": "Agent-managed graph exploration",
                "confidence_score": 1.0,
                "tool_calls": result.get("tool_calls", []),
                "context": get_text_context() or result.get("context", "") or "Agent exploration context",
            }
        except Exception as e:
            logger.error("Agent endpoint error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # Fall back to direct retrieval pipeline
    config = get_app_config()
    db_type = request.database

    try:
        db: GraphDB = get_database_client(config, db_type)
        try:
            accumulated_graph = request.accumulated_graph_data
            if accumulated_graph and not isinstance(accumulated_graph, dict):
                accumulated_graph = None

            # For now, we reuse the existing run_focused_retrieval service.
            result = run_focused_retrieval(
                db,
                query=request.query,
                messages=request.messages,
                accumulated_graph=accumulated_graph,
            )

            new_nodes_count = len(result.graph_data.get("nodes", []))
            new_edges_count = len(result.graph_data.get("edges", []))
            if accumulated_graph:
                existing_nodes_count = len(accumulated_graph.get("nodes", []))
                existing_edges_count = len(accumulated_graph.get("edges", []))
                new_nodes_count = new_nodes_count - existing_nodes_count
                new_edges_count = new_edges_count - existing_edges_count

            reasoning_chain = [
                f"Database: {db_type}",
                f"Extracted keywords: {', '.join(result.keywords)}",
                f"Identified seed entities: {', '.join(result.seed_entities)}",
                f"Expanded subgraph: {new_nodes_count} new nodes, {new_edges_count} new edges",
                f"Total accumulated: {len(result.graph_data.get('nodes', []))} nodes, {len(result.graph_data.get('edges', []))} edges",
                f"Context length: {len(result.context)} chars",
                f"Response time: {result.execution_time:.2f}s",
            ]

            # Append graph stats and memory diagnostics to reasoning/debug info when available
            if getattr(result, "graph_stats", None):
                gs = result.graph_stats or {}
                reasoning_chain.append(
                    "Graph stats: "
                    f"{gs.get('nodes', 0)} nodes, "
                    f"{gs.get('relationships', 0)} relationships, "
                    f"FalkorDB memory: {gs.get('falkordb_memory_human', 'N/A')}, "
                    f"Python RSS: {gs.get('python_process_memory_mb', 'N/A')} MB"
                )

            if getattr(result, "query_memory_mb", None):
                qm = result.query_memory_mb or {}
                reasoning_chain.append(
                    "Query memory usage: "
                    f"before={qm.get('before_mb', 'N/A')} MB, "
                    f"peak={qm.get('peak_mb', 'N/A')} MB, "
                    f"delta={qm.get('delta_mb', 'N/A')} MB"
                )

            # Detailed timing breakdown
            timings = getattr(result, "detailed_timing", {})
            if timings:
                seed_keys = ["pgvector_topic", "pgvector_subtopic", "falkordb_entity_vector", "falkordb_keyword", "seed_reranking"]
                expansion_keys = [k for k in timings.keys() if k.startswith("expand_")]
                
                seed_lines = []
                for k in seed_keys:
                    if k in timings:
                        display_name = k.replace("_", " ").title().replace("Pgvector", "Pgvector").replace("Falkordb", "FalkorDB")
                        seed_lines.append(f"  • {display_name}: {timings[k]:.3f}s")
                
                expansion_lines = []
                processed_exp = {}
                for k in expansion_keys:
                    simple_key = k.replace("expand_", "").replace("_", " ").title()
                    # Aggregate batches
                    if "Batch" in simple_key:
                        base = simple_key.split(" Batch")[0]
                        processed_exp[base] = processed_exp.get(base, 0.0) + timings[k]
                    else:
                        processed_exp[simple_key] = timings[k]
                
                # Sort expansion lines? maybe by value or name? Let's just keep dict order or sort by name
                for k in sorted(processed_exp.keys()):
                    expansion_lines.append(f"  • {k}: {processed_exp[k]:.3f}s")
                
                if seed_lines:
                    reasoning_chain.append("⏱️ Seed Identification:\n" + "\n".join(seed_lines))
                if expansion_lines:
                    reasoning_chain.append("⏱️ Subgraph Expansion:\n" + "\n".join(expansion_lines))

            return {
                "answer": result.answer,
                "context": result.context,
                "full_prompt": result.full_prompt,
                "execution_time": result.execution_time,
                "graph_data": result.graph_data,
                "graph_stats": getattr(result, "graph_stats", None),
                "query_memory_mb": getattr(result, "query_memory_mb", None),
                "reasoning_chain": reasoning_chain,
                "cypher_query": "Dynamic retrieval based on keywords",
                "confidence_score": 1.0,
            }
        finally:
            db.close()

    except Exception as e:  # noqa: BLE001
        logger.error("Endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat")
async def agent_chat_endpoint(request: ChatRequest):
    """
    Dedicated endpoint for LlamaIndex agent-based chat.
    
    Always uses the proactive LlamaIndex agent that explores the graph
    before answering questions.
    """
    import time
    start_time = time.time()
    
    try:
        # Initialize graph context with any accumulated data from previous turns
        init_graph_context(request.accumulated_graph_data)
        
        agent = get_agent(verbose=False)
        
        # Convert messages to history format
        history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Run agent
        result = await agent.chat(request.query, history)
        
        execution_time = time.time() - start_time
        
        # Get accumulated graph data from tools used during this turn
        graph_data = get_accumulated_data()
        
        return {
            "answer": result["answer"],
            "execution_time": execution_time,
            "tool_calls": result.get("tool_calls", []),
            "iterations": result.get("iterations", 0),
            "graph_data": graph_data,
            # Context is managed internally by the agent, but we can return graph stats or summary if needed
            "reasoning_chain": result.get("reasoning_timeline", []),
        }
    except Exception as e:
        logger.error("Agent chat endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/node-connections/{node_id}")
async def get_node_connections(node_id: str, database: str = "falkordb"):
    """
    Fetch all immediate connections (neighbors) of a given node.
    Returns nodes and edges connected to the specified node.
    """
    config = get_app_config()
    try:
        db: GraphDB = get_database_client(config, database)
        try:
            # Query to get all connected nodes and edges
            # Use id() for compatibility with FalkorDB
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



