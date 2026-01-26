from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# GraphRAG Service URL
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag:8000")

class RetrievalRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class RetrievalResponse(BaseModel):
    keywords: List[str] = []
    seeds: List[str] = []
    subgraph_nodes: List[Dict[str, Any]] = []
    subgraph_edges: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}
    answer: str
    context: str
    reasoning_chain: List[str] = []

@router.post("/debug", response_model=RetrievalResponse)
async def debug_retrieval(req: RetrievalRequest):
    """
    Proxy the retrieval request to the GraphRAG service.
    """
    try:
        logger.info(f"Sending query to GraphRAG at {GRAPHRAG_URL}: {req.query}")
        
        # Call GraphRAG service
        response = requests.post(
            f"{GRAPHRAG_URL}/chat",
            json={
                "query": req.query,
                "create_plot": False, # We can enable this if we want graph data
                # Add other params if needed
            },
            timeout=120
        )
        
        if response.status_code != 200:
            logger.error(f"GraphRAG error {response.status_code}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"GraphRAG service error: {response.text}")
            
        data = response.json()
        
        # Map GraphRAG response to Dashboard response format
        # GraphRAG returns: 
        # {
        #     "answer": ...,
        #     "context": ...,
        #     "graph_data": {"nodes": [], "edges": []},
        #     "execution_time": ...,
        #     "reasoning_chain": [],
        #     "seed_entities": [],
        #     ...
        # }
        
        graph_data = data.get("graph_data", {})
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Format nodes/edges if necessary for frontend
        # Assuming frontend expects them as-is or we might need transformation
        
        return RetrievalResponse(
            keywords=[], # GraphRAG might not expose keywords directly in top level
            seeds=data.get("seed_entities", []),
            subgraph_nodes=nodes,
            subgraph_edges=edges,
            timings={"total": data.get("execution_time", 0)},
            answer=data.get("answer", ""),
            context=str(data.get("context", "")),
            reasoning_chain=data.get("reasoning_chain", [])
        )
        
    except requests.RequestException as e:
        logger.error(f"Connection to GraphRAG failed: {e}")
        raise HTTPException(status_code=503, detail=f"GraphRAG service unreachable: {str(e)}")
    except Exception as e:
        logger.error(f"Retrieval proxy failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
