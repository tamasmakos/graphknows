from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.services.retrieval import (
    extract_keywords,
    get_seed_entities,
    expand_subgraph,
    get_graph_stats,
    format_graph_context
)
from src.app.infrastructure.graph_db import GraphDB
from src.app.infrastructure.llm import get_llm, get_embedding_model
from ..database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

class RetrievalRequest(BaseModel):
    query: str

class RetrievalResponse(BaseModel):
    keywords: List[str]
    seeds: List[str]
    subgraph_nodes: List[Dict[str, Any]]
    subgraph_edges: List[Dict[str, Any]]
    timings: Dict[str, float]
    answer: str
    context: str

@router.post("/debug", response_model=RetrievalResponse)
async def debug_retrieval(req: RetrievalRequest, db: GraphDB = Depends(get_db)):
    """
    Run the retrieval pipeline for a given query and return intermediate results + answer.
    """
    start_total = time.time()
    timings = {}
    
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")

    # 1. Initialize Components
    try:
        # db is injected
        llm = get_llm()
        embed_model = get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to init retrieval components: {e}")
        raise HTTPException(status_code=500, detail=f"Component init failed: {str(e)}")

    try:
        # 2. Extract Keywords
        t0 = time.time()
        keywords = extract_keywords(llm, req.query)
        timings["keywords"] = time.time() - t0
        
        # 3. Embed Query
        t0 = time.time()
        embedding = embed_model.embed_query(req.query)
        timings["embedding"] = time.time() - t0
        
        # 4. Get Seeds
        t0 = time.time()
        seeds, seed_timings = get_seed_entities(db, embedding, keywords)
        timings["seed_finding"] = time.time() - t0
        timings.update(seed_timings)
        
        # 5. Expand Subgraph
        t0 = time.time()
        raw_nodes, raw_edges, expand_timings = expand_subgraph(db, seeds)
        timings["subgraph_expansion"] = time.time() - t0
        timings.update(expand_timings)
        
        # 6. Format Context
        nodes_dict = {
            node.get("element_id") or node.get("id"): node for node in raw_nodes.values()
        }
        context = format_graph_context(nodes_dict, raw_edges)

        # 7. Generate Answer
        t0 = time.time()
        system_prompt = """You are a Personal Life Assistant.
        
        **Available Information:**
        {context}
        """
        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=req.query)
        ]
        response = llm.invoke(messages)
        answer = response.content
        timings["llm_generation"] = time.time() - t0

        # 8. Transform for Frontend
        id_map = { eid: n_data["id"] for eid, n_data in raw_nodes.items() }
        
        nodes_list = []
        for eid, n_data in raw_nodes.items():
            nodes_list.append(n_data)
            
        edges_list = []
        for edge in raw_edges:
            start_display = id_map.get(str(edge["start"]))
            end_display = id_map.get(str(edge["end"]))
            
            if start_display and end_display:
                edges_list.append({
                    "id": f"{start_display}-{end_display}-{edge['type']}",
                    "source": start_display,
                    "target": end_display,
                    "type": edge["type"],
                    "properties": edge["properties"]
                })
        
        timings["total"] = time.time() - start_total
        
        return {
            "keywords": keywords,
            "seeds": seeds,
            "subgraph_nodes": nodes_list,
            "subgraph_edges": edges_list,
            "timings": timings,
            "answer": answer,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Retrieval pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
    finally:
        pass