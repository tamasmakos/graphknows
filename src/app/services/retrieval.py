"""
Retrieval pipeline services for the Knowledge Graph backend.

This module contains the core graph-retrieval logic used by both the HTTP API
and the MCP server, independent of any particular interface framework.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import resource

from pydantic import BaseModel

from src.app.infrastructure.graph_db import GraphDB
from src.app.infrastructure.llm import get_llm

logger = logging.getLogger(__name__)


MAX_EXPANDED_NODES = 300
MAX_EXPANDED_RELATIONSHIPS = 500
MAX_ENTITY_HOP = 2
MAX_ENTITY_NEIGHBORS_PER_SEED = 50
MAX_RELATIONSHIPS_PER_SEED = 100


class Profiler:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        msg = f"[Profiler] {self.name} took {duration:.3f}s"
        logger.info(msg)
        print(msg, flush=True)  # Force output to stdout for visibility


class Message(BaseModel):
    role: str
    content: str


class QueryResult(BaseModel):
    answer: str
    context: str
    execution_time: float
    keywords: List[str] = []
    graph_data: dict = {}
    seed_entities: List[str] = []
    graph_stats: Optional[Dict[str, Any]] = None
    query_memory_mb: Optional[Dict[str, float]] = None
    full_prompt: Optional[str] = None
    detailed_timing: Dict[str, float] = {}


def clean_entity_name(name: str) -> str:
    """
    Clean entity names by removing numeric prefixes that might have been
    introduced during extraction (e.g., '12 fideszkdnp' -> 'fideszkdnp').
    """
    if not name or not isinstance(name, str):
        return name
    
    # Remove leading numbers followed by space (e.g. "12 Name")
    # But preserve names that are just numbers or start with numbers but are short/specific?
    # For now, strict pattern: start of string, one or more digits, one space.
    import re
    return re.sub(r'^\d+\s+', '', name)


def get_query_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "before_mb": round(memory_info.rss / 1024 / 1024, 1),
    }


def get_graph_stats(db: GraphDB) -> Dict[str, Any]:
    """Get graph statistics including node counts, relationship counts, and memory info."""
    stats = {
        "nodes": 0,
        "relationships": 0,
        "falkordb_memory_human": "N/A",
        "python_process_memory_mb": "N/A",
    }
    
    try:
        # Get node count
        node_result = db.query("MATCH (n) RETURN count(n) as count")
        if node_result:
            stats["nodes"] = node_result[0].get("count", 0)
        
        # Get relationship count
        rel_result = db.query("MATCH ()-[r]->() RETURN count(r) as count")
        if rel_result:
            stats["relationships"] = rel_result[0].get("count", 0)
        
        # Get Python process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        stats["python_process_memory_mb"] = round(memory_info.rss / 1024 / 1024, 1)
        
        # Try to get FalkorDB memory info if it's a FalkorDB instance
        if hasattr(db, 'graph') and hasattr(db.graph, 'config'):
            try:
                # FalkorDB memory info query
                mem_result = db.query("CALL db.info()")
                if mem_result and isinstance(mem_result, list) and len(mem_result) > 0:
                    # Try to extract memory info from the result
                    for record in mem_result:
                        if isinstance(record, dict):
                            # Look for memory-related keys
                            for key, value in record.items():
                                if 'memory' in key.lower() or 'mem' in key.lower():
                                    if isinstance(value, (int, float)):
                                        # Convert bytes to human-readable format
                                        if value > 1024 * 1024 * 1024:
                                            stats["falkordb_memory_human"] = f"{value / (1024**3):.2f} GB"
                                        elif value > 1024 * 1024:
                                            stats["falkordb_memory_human"] = f"{value / (1024**2):.2f} MB"
                                        else:
                                            stats["falkordb_memory_human"] = f"{value / 1024:.2f} KB"
                                    else:
                                        stats["falkordb_memory_human"] = str(value)
                                    break
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not retrieve FalkorDB memory info: %s", e)
        
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to get graph stats: %s", e)
    
    return stats


def extract_keywords(llm, query: str) -> List[str]:
    """Extract keywords/entities from the user query."""
    from langchain_core.prompts import ChatPromptTemplate
    import re

    prompt = ChatPromptTemplate.from_template(
        """
    Extract the key entities and important terms from the following user query.
    Return ONLY a JSON object with a "keywords" key containing a list of strings.
    Do not include generic terms like "what", "who", "where", "tell me about".
    Focus on specific names, locations, concepts, organizations, and key adjectives/attributes.

    Query: {query}
    
    JSON Output:
    """
    )

    chain = prompt | llm
    try:
        with Profiler("LLM Keyword Extraction"):
            result = chain.invoke({"query": query})
        
        # Extract content from response
        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)
        
        # Strip <think>...</think> tags (reasoning model output)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        # Try to parse JSON from cleaned content
        try:
            parsed = json.loads(content)
            return parsed.get("keywords", [])
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            json_match = re.search(r'\{.*?\}', content, flags=re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed.get("keywords", [])
            logger.warning("Could not parse JSON from LLM output: %s", content[:200])
            return []
            
    except Exception as e:  # noqa: BLE001
        logger.error("Keyword extraction failed: %s", e)
        return []



def get_seed_entities(
    db: GraphDB, query_embedding: List[float], keywords: List[str]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Identify and rerank seed entities using vector search and keyword matching.
    Optimized to run searches in parallel.
    Returns:
        List of seed entities
        Dictionary of timing stats
    """
    candidates: Dict[str, float] = {}  # name -> score
    timings: Dict[str, float] = {}

    def search_topic():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            # This uses pgvector if available
            topic_results = db.query_vector(
                "TOPIC", query_embedding, k=3, min_score=0.55
            )
            for node_data, vector_score in topic_results:
                # Instead of relying on entity_ids property, we expand in the graph
                topic_id = node_data.get("id")
                if topic_id:
                     # Find entities in this topic
                     cypher = """
                     MATCH (t:TOPIC)-[:IN_TOPIC]-(e)
                     WHERE t.id = $id
                     RETURN e.id as id, coalesce(e.pagerank_centrality, 0.0) as pr_score
                     LIMIT 20
                     """
                     connected = db.query(cypher, {'id': topic_id})
                     for row in connected:
                         eid = row.get('id')
                         pr = row.get('pr_score', 0)
                         # Score is mix of vector match and static centrality
                         combined_score = vector_score * 0.5 + pr * 0.5
                         local_candidates[eid] = max(local_candidates.get(eid, 0), combined_score)

        except Exception as e:  # noqa: BLE001
            logger.warning("Topic search failed: %s", e)
        
        return local_candidates, time.time() - t0

    def search_subtopic():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            # This uses pgvector if available
            subtopic_results = db.query_vector(
                "SUBTOPIC", query_embedding, k=3, min_score=0.55
            )
            for node_data, vector_score in subtopic_results:
                subtopic_id = node_data.get("id")
                if subtopic_id:
                     # Find entities in this subtopic
                     cypher = """
                     MATCH (t:SUBTOPIC)-[:IN_TOPIC]-(e)
                     WHERE t.id = $id
                     RETURN e.id as id, coalesce(e.pagerank_centrality, 0.0) as pr_score
                     LIMIT 20
                     """
                     connected = db.query(cypher, {'id': subtopic_id})
                     for row in connected:
                         eid = row.get('id')
                         pr = row.get('pr_score', 0)
                         combined_score = vector_score * 0.5 + pr * 0.5
                         local_candidates[eid] = max(local_candidates.get(eid, 0), combined_score)
                         
        except Exception as e:  # noqa: BLE001
            logger.warning("Subtopic search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_entity():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            # This generally uses FalkorDB vector index unless configured otherwise
            entity_results = db.query_vector(
                "ENTITY_CONCEPT", query_embedding, k=10, min_score=0.5
            )
            for node_data, score in entity_results:
                # Use ID as primary key
                entity_id = node_data.get("id")
                if entity_id:
                    local_candidates[entity_id] = max(local_candidates.get(entity_id, 0), score)
        except Exception as e:  # noqa: BLE001
            logger.warning("Entity vector search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_keywords():
        t0 = time.time()
        local_candidates = {}
        if not keywords:
            return local_candidates, 0.0
        try:
            # This uses FalkorDB Cypher
            with Profiler("Search Keywords"):
                keyword_query = """
                UNWIND $keywords AS keyword
                MATCH (seed)
                WHERE (seed:Entity OR seed:ENTITY_CONCEPT OR seed:Topic OR seed:Subtopic OR seed:TOPIC OR seed:SUBTOPIC OR seed:DAY OR seed:SEGMENT OR seed:CONVERSATION OR seed:Day OR seed:Segment OR seed:Conversation)
                AND (
                    toLower(coalesce(seed.name, "")) CONTAINS toLower(keyword)
                    OR toLower(coalesce(seed.id, "")) CONTAINS toLower(keyword)
                    OR toLower(coalesce(seed.title, "")) CONTAINS toLower(keyword)
                )
                RETURN DISTINCT seed.id AS id
                LIMIT 100
                """
                results = db.query(keyword_query, {"keywords": keywords})
            for record in results:
                entity_id = record.get("id")
                if entity_id:
                    local_candidates[entity_id] = max(local_candidates.get(entity_id, 0), 1.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("Keyword search failed: %s", e)
        return local_candidates, time.time() - t0

    # Run searches in parallel
    with Profiler("Parallel Seed Search"), ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(search_topic): "pgvector_topic",
            executor.submit(search_subtopic): "pgvector_subtopic",
            executor.submit(search_entity): "falkordb_entity_vector",
            executor.submit(search_keywords): "falkordb_keyword",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                result, duration = future.result()
                timings[key] = duration
                for entity, score in result.items():
                    candidates[entity] = max(candidates.get(entity, 0), score)
            except Exception as e:  # noqa: BLE001
                logger.error("Search %s failed: %s", key, e)
                timings[key] = 0.0

    if not candidates:
        return [], timings

    # 4. Rerank (combine with PageRank)
    final_seeds: List[str] = []
    rerank_t0 = time.time()
    try:
        with Profiler("Seed Reranking"):
            rerank_query = """
            UNWIND $ids AS id
            MATCH (e)
            WHERE e.id = id
            RETURN e.id AS id, coalesce(e.pagerank_centrality, e.pagerank, 0.0) AS pagerank
            """
            results = db.query(rerank_query, {"ids": list(candidates.keys())})

        scored_entities: List[Tuple[str, float]] = []
        for record in results:
            name = record.get("id")
            pagerank = record.get("pagerank", 0.0)
            base_score = candidates.get(name, 0)

            final_score = (base_score * 0.7) + (pagerank * 0.3)
            scored_entities.append((name, final_score))

        scored_entities.sort(key=lambda x: x[1], reverse=True)
        final_seeds = [x[0] for x in scored_entities[:MAX_ENTITY_NEIGHBORS_PER_SEED]]
        logger.info("Top seeds after reranking: %s", final_seeds)

    except Exception as e:  # noqa: BLE001
        logger.error("Reranking failed: %s", e)
        final_seeds = list(candidates.keys())[:5]
    timings["seed_reranking"] = time.time() - rerank_t0

    return final_seeds, timings


# The rest of the helpers are largely copied from the existing backend module.
# They are left unchanged in behavior but centralized here.


def filter_node_properties(props: dict, labels: List[str]) -> dict:
    """
    Filter node properties to only include distilled, human-readable information.
    Removes raw pipeline outputs, embeddings, internal metadata, and intermediate calculations.
    """
    if not props:
        return {}

    labels_lower = {label.lower() for label in labels} if labels else set()

    is_day = "day" in labels_lower
    is_segment = "segment" in labels_lower
    is_chunk = "chunk" in labels_lower
    is_entity = "entity" in labels_lower or "entity_concept" in labels_lower
    is_topic = "topic" in labels_lower or (
        props.get("title")
        and props.get("summary")
        and props.get("community_id")
        and not props.get("name")
    )
    is_subtopic = "subtopic" in labels_lower or (
        props.get("title")
        and props.get("summary")
        and (props.get("subtopic_local_id") or props.get("parent_topic"))
        and not props.get("name")
    )
    is_community = "community" in labels_lower and not is_topic and not is_subtopic
    is_community = "community" in labels_lower and not is_topic and not is_subtopic
    is_subcommunity = "subcommunity" in labels_lower
    is_conversation = "conversation" in labels_lower

    filtered: Dict[str, Any] = {}

    always_exclude = {
        "embedding",
        "text_embedding",
        "kge_embedding",
        "graph_type",
        "element_id",
        "id",
        "node_type",
        "raw_extraction",
        "extraction_successful",
        "has_summary",
        "updated_at",
        "community",
        "chunk_ids",
        "entity_ids",
        "rdf_type",
        "foaf_name",
        "foaf_firstName",
        "foaf_lastName",
        "foaf_gender",
        "bio_birth",
        "Text_ID",
        "Title",
        "Date",
        "Body",
        "Term",
        "Party_status",
        "degree_centrality_distance_from_mean",
        "degree_centrality_deviation_from_mean",
        "degree_centrality_z_score",
        "betweenness_centrality_distance_from_mean",
        "betweenness_centrality_deviation_from_mean",
        "betweenness_centrality_z_score",
        "closeness_centrality_distance_from_mean",
        "closeness_centrality_deviation_from_mean",
        "closeness_centrality_z_score",
        "eigenvector_centrality_distance_from_mean",
        "eigenvector_centrality_deviation_from_mean",
        "eigenvector_centrality_z_score",
        "pagerank_centrality_distance_from_mean",
        "pagerank_centrality_deviation_from_mean",
        "pagerank_centrality_z_score",
        "harmonic_centrality_distance_from_mean",
        "harmonic_centrality_deviation_from_mean",
        "harmonic_centrality_z_score",
        "load_centrality_distance_from_mean",
        "load_centrality_deviation_from_mean",
        "load_centrality_z_score",
    }

    if is_day:
        keep_keys = {
            "date",
            "name",
            "episode_count",
            "segment_count",
            "document_date",  # Also preserve document_date if present
        }
    elif is_segment:
        keep_keys = {
            "content",
            "date",
            "document_date",
            "sentiment",
            "name",
            "line_number",
        }
    elif is_chunk:
        keep_keys = {
            "text",
            "llama_metadata",
            "knowledge_triplets",
        }
    elif is_entity:
        keep_keys = {
            "name",
            "entity_type",
            "centrality_summary",
            "pagerank_centrality",
            # "degree_centrality_description",
            # "betweenness_centrality_description",
            # "closeness_centrality_description",
            # "eigenvector_centrality_description",
            # "pagerank_centrality_description",
        }
    elif is_topic or is_community:
        keep_keys = {
            "title",
            "summary",
            "community_id",
        }
    elif is_subtopic or is_subcommunity:
        keep_keys = {
            "title",
            "summary",
            "community_id",
        }
    elif is_conversation:
        keep_keys = {
            "time",
            "name",
            "location",
            "date", # if present
        }
    else:
        keep_keys = {
            "name",
            "title",
            "summary",
        }

    for key, value in props.items():
        if key in always_exclude:
            continue
        if key in keep_keys:
            filtered[key] = value
        elif key not in always_exclude and not any(
            ex in key.lower()
            for ex in [
                "_id",
                "_idx",
                "_embedding",
                "_z_score",
                "_distance",
                "_deviation",
            ]
        ):
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value

    return filtered


def _process_graph_results(results: List[Dict[str, Any]], nodes: Dict[str, Any], edges: List[Dict[str, Any]]):
    """
    Helper to process query results and update nodes/edges collections.
    Handles FalkorDB result formats and property extraction.
    """
    for record in results:
        # Process Nodes
        for key, value in record.items():
            # Check if value looks like a Node (has properties/labels or is dict with them)
            # Simple heuristic: inspect keys or attributes
            candidate_node = None
            if hasattr(value, "labels") and hasattr(value, "id"): # Object
                candidate_node = value
            elif isinstance(value, dict) and ("labels" in value or "properties" in value): # Dict representation
                candidate_node = value
            
            if candidate_node:
                node_props: Dict[str, Any] = {}
                labels: List[str] = []
                element_id = ""

                if hasattr(candidate_node, "properties"):
                    raw_props = candidate_node.properties
                    if hasattr(raw_props, "items"):
                        node_props = dict(raw_props)
                    elif isinstance(raw_props, dict):
                        node_props = raw_props.copy()
                    else:
                        node_props = {}

                    if hasattr(candidate_node, "labels"):
                        raw_labels = candidate_node.labels
                        if isinstance(raw_labels, (list, tuple)):
                            labels = list(raw_labels)
                        elif raw_labels:
                            labels = [raw_labels]
                    
                    if hasattr(candidate_node, "id"):
                        element_id = str(candidate_node.id)
                    elif hasattr(candidate_node, "element_id"):
                        element_id = str(candidate_node.element_id)
                    else:
                        element_id = str(node_props.get("id") or "unknown")

                elif isinstance(candidate_node, dict):
                    node_props = candidate_node.get("properties", {}).copy()
                    if not node_props and "properties" not in candidate_node:
                         # It might be a flat dict of properties if it came from certain queries
                         # But typically our query returns nodes. 
                         # If it's the dict structure from fallback:
                         node_props = candidate_node.copy()
                         node_props.pop("labels", None)
                         node_props.pop("id", None)
                         node_props.pop("element_id", None)

                    labels_raw = candidate_node.get("labels", [])
                    if isinstance(labels_raw, str):
                        labels = [labels_raw]
                    else:
                        labels = list(labels_raw)
                    
                    element_id = str(candidate_node.get("id") or candidate_node.get("element_id") or "")
                
                # Check for nested JSON strings
                for k, v in list(node_props.items()):
                    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                        try:
                            node_props[k] = json.loads(v)
                        except Exception: # noqa: BLE001
                            pass

                filtered_props = filter_node_properties(node_props, labels)
                
                node_display_id = (
                    filtered_props.get("name")
                    or filtered_props.get("title")
                    or node_props.get("id")
                    or element_id
                )

                nodes[element_id] = {
                    "id": node_display_id,
                    "element_id": element_id,
                    "labels": labels,
                    "properties": filtered_props,
                }

        # Process Relationships
        # Look for relationship objects or specific keys like 'r', 'rel', etc.
        # But we iterate all values to be generic
        for key, value in record.items():
            candidate_rel = None
            if hasattr(value, "start_node") and hasattr(value, "end_node"): # Object
                candidate_rel = value
            elif hasattr(value, "src_node") and hasattr(value, "dest_node"): # FalkorDB Object
                candidate_rel = value
            elif isinstance(value, dict) and "start" in value and "end" in value and "type" in value: # Dict
                candidate_rel = value
            
            if candidate_rel:
                rel_props: Dict[str, Any] = {}
                rel_type = "RELATED"
                start_id = ""
                end_id = ""

                if hasattr(candidate_rel, "properties"):
                    raw_props = candidate_rel.properties
                    rel_props = dict(raw_props) if hasattr(raw_props, "items") else {}
                    rel_props = dict(raw_props) if hasattr(raw_props, "items") else {}
                    # FalkorDB Edge uses 'relation', others 'type'
                    if hasattr(candidate_rel, "relation"):
                         rel_type = candidate_rel.relation
                    else:
                         rel_type = getattr(candidate_rel, "type", "RELATED")
                    
                    if hasattr(candidate_rel, "start_node"):
                        start_id = str(getattr(candidate_rel.start_node, "id", "")) or str(getattr(candidate_rel.start_node, "element_id", ""))
                        end_id = str(getattr(candidate_rel.end_node, "id", "")) or str(getattr(candidate_rel.end_node, "element_id", ""))
                    elif hasattr(candidate_rel, "src_node"): # RedisGraph client sometimes
                         start_id = str(candidate_rel.src_node)
                         end_id = str(candidate_rel.dest_node)

                elif isinstance(candidate_rel, dict):
                    rel_props = candidate_rel.get("properties", {}).copy()
                    if not rel_props:
                         rel_props = candidate_rel.copy()
                         rel_props.pop("start", None)
                         rel_props.pop("end", None)
                         rel_props.pop("type", None)
                    
                    rel_type = candidate_rel.get("type", "RELATED")
                    start_id = str(candidate_rel.get("start", ""))
                    end_id = str(candidate_rel.get("end", ""))

                if start_id and end_id:
                     # Check if we have these nodes
                    # Only add edge if we have both nodes? NO, we might process nodes in other batches.
                    # Just add edge for now.
                    edges.append({
                        "start": start_id,
                        "end": end_id,
                        "type": rel_type,
                        "properties": rel_props
                    })

def expand_subgraph(
    db: GraphDB, seed_entities: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    """
    Perform a split, multi-stage traversal to avoid DB timeouts.
    Stages:
    1. Fetch Seed Nodes
    2. Fetch Chunks connected to Seeds
    3. Fetch Content Hierarchy (Chunk->Segment->Day)
    4. Fetch Context Neighbors (Chunk->Entity)
    
    Returns:
        nodes: Dict[str, Any]
        edges: List[Dict[str, Any]]
        timings: Dict[str, float]
    """
    nodes: Dict[str, Any] = {}
    edges: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}

    if not seed_entities:
        return nodes, edges, timings

    def run_stage(name: str, cypher: str, params: dict):
        t0 = time.time()
        try:
            with Profiler(f"Expand {name}"):
                result = db.query(cypher, params)
                _process_graph_results(result, nodes, edges)
                logger.info(f"{name}: Fetched {len(result)} records")
        except Exception as e:
            logger.error(f"Expansion Stage {name} failed: {e}")
        finally:
            timings[f"expand_{name.lower().replace(' ', '_')}"] = time.time() - t0

    try:
        # 1. Fetch Seed Nodes
        run_stage("Seed Nodes", """
            UNWIND $seeds AS seed_id
            MATCH (n)
            WHERE n.id = seed_id
            RETURN n
            """, {"seeds": seed_entities})
        
        # Identify IDs for next steps
        seed_ids = [int(n_id) for n_id in nodes.keys() if n_id.isdigit()]
        if not seed_ids:
             # Try assuming string IDs if not digit
             seed_ids = [n for n in nodes.keys()]
        
        # 6. Fetch DIRECT relationships for SEED entities (Semantic Expansion)
        # The user wants to see edges like "battery factory" SUPPORTS "government", even if government wasn't in chunks.
        # We expand 1st degree connections for seeds to enrich the graph.
        
        logger.info(f"Expanding semantic relationships for {len(seed_ids)} seeds...")
        seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
        
        # Query: Seed -[r]- Entity
        # Limit per seed to avoid explosion
        semantic_query = f"""
        MATCH (s)-[r]-(e)
        WHERE ID(s) IN {seed_ids_str}
        AND (labels(e) = ['Entity'] OR 'Entity' IN labels(e) OR labels(e) = ['ENTITY_CONCEPT'] OR 'ENTITY_CONCEPT' IN labels(e))
        AND type(r) <> 'HAS_ENTITY'
        RETURN s, r, e
        LIMIT 200
        """
        # Note: directionless match to catch valid semantic edges
        run_stage("Semantic Expansion", semantic_query, {})
        logger.info(f"Semantic Expansion found {len(nodes) - len(seed_entities)} new nodes.") # Approximation
        
        # 7. Fetch DIRECT relationships between ALL found entities (Closure)
        # Collect all entity IDs found so far (Seeds + Neighbors + Newly added semantic ones)
        all_entity_ids = []
        for nid, n in nodes.items():
            labels = getattr(n, 'labels', [])
            if hasattr(n, 'labels') and ( 'Entity' in labels or 'ENTITY_CONCEPT' in labels ):
                all_entity_ids.append(nid)
            elif isinstance(n, dict): # Fallback if dict
                labels = n.get('labels', [])
                if 'Entity' in labels or 'ENTITY_CONCEPT' in labels:
                    all_entity_ids.append(nid)
        
        # Add seeds to be sure
        for sid in seed_ids:
            if sid not in all_entity_ids:
                all_entity_ids.append(sid)

        # Batch this N*N check (matching on set)
        if all_entity_ids:
             all_entity_ids = list(set(all_entity_ids))
             batch_size = 500
             for i in range(0, len(all_entity_ids), batch_size):
                 batch = all_entity_ids[i : i + batch_size]
                 batch_str = "[" + ", ".join(str(eid) for eid in batch) + "]"
                 
                 rel_enrich_query = f"""
                 MATCH (a)-[r]->(b)
                 WHERE ID(a) IN {batch_str}
                 AND ID(b) IN {batch_str}
                 RETURN a, r, b
                 """
                 run_stage(f"Entity Closure Batch {i//batch_size}", rel_enrich_query, {})

        # Structure the result
        if not seed_ids:
            return nodes, edges, timings

        logger.info("Found %d seed nodes. expanding chunks...", len(seed_ids))

        # 2. Fetch Connected Chunks
        # Limit per seed to avoid explosion
        # Manually interpolate IDs because FalkorDB/RedisGraph param binding for ID lists can be flaky
        seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
        
        chunk_query = f"""
        MATCH (s)
        WHERE ID(s) IN {seed_ids_str}
        OPTIONAL MATCH (s)<-[r1:HAS_ENTITY]-(c1)
        OPTIONAL MATCH (s)-[r2:HAS_CHUNK]->(c2)
        RETURN s, r1, c1, r2, c2
        LIMIT 200
        """
        run_stage("Connected Chunks", chunk_query, {})
        
        chunk_ids = [
            int(n["element_id"]) 
            for n in nodes.values() 
            if "Chunk" in n.get("labels", []) or n.get("properties", {}).get("text")
        ]
        
        logger.info("Found %d relevant chunks. expanding hierarchy...", len(chunk_ids))
        
        if chunk_ids:
            # 3. Fetch Hierarchy (Day -> Segment -> Chunk)
            # Traverse strict path: Day -> Segment -> Chunk
            batch_size = 100
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                batch_str = "[" + ", ".join(str(cid) for cid in batch) + "]"
                
                # Hierarchy Query: Handle both direct Segment->Chunk and Segment->Conversation->Chunk paths
                # Also fetch Place and Context connected to Conversation
                hierarchy_query = f"""
                MATCH (c) WHERE ID(c) IN {batch_str}
                OPTIONAL MATCH (c)<-[r1a:HAS_CHUNK]-(seg:SEGMENT)<-[r2a:HAS_SEGMENT]-(day:DAY)
                OPTIONAL MATCH (c)<-[r1b:HAS_CHUNK]-(conv:CONVERSATION)
                OPTIONAL MATCH (conv)<-[r2b:HAS_CONVERSATION]-(seg2:SEGMENT)<-[r3b:HAS_SEGMENT]-(day2:DAY)
                OPTIONAL MATCH (conv)-[r_place:HAPPENED_AT]->(place:PLACE)
                OPTIONAL MATCH (conv)-[r_ctx:HAS_CONTEXT]->(context:CONTEXT)
                RETURN c, r1a, seg, r2a, day, r1b, conv, r2b, seg2, r3b, day2, r_place, place, r_ctx, context
                """
                run_stage(f"Hierarchy Batch {i//batch_size}", hierarchy_query, {})

            # 4. Fetch Context Neighbors (Chunk -> other Entities)
            # Limit to high relevance, avoid exploding the graph
            # User requested "first degree entity neighbors"
            seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
            for i in range(0, len(chunk_ids), batch_size):
                 batch = chunk_ids[i : i + batch_size]
                 batch_str = "[" + ", ".join(str(cid) for cid in batch) + "]"
                 
                 # Only fetch entities that are NOT the seeds (to avoid self-loops/duplication in logic)
                 neighbor_query = f"""
                 MATCH (c)-[r:HAS_ENTITY]->(e)
                 WHERE ID(c) IN {batch_str}
                 AND NOT ID(e) IN {seed_ids_str}
                 RETURN c, r, e
                 LIMIT 200
                 """
                 run_stage(f"Context Neighbors Batch {i//batch_size}", neighbor_query, {})

        # 5. Fetch Topic Hierarchy (Entity -> Subtopic -> Topic)
        # This was missing in previous iteration.
        logger.info("Expanding topic hierarchy for seeds...")
        seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
        
        # Simplified query if labels are consistent in DB (e.g., all uppercase based on schema check)
        # Schema check showed ['TOPIC'] and ['SUBTOPIC']
        topic_query = f"""
        MATCH (e)-[r1]-(s)-[r2]-(t)
        WHERE ID(e) IN {seed_ids_str}
        AND (type(r1) = 'IN_TOPIC' OR type(r1) = 'in_topic') 
        AND (labels(s) = ['SUBTOPIC'] OR 'SUBTOPIC' IN labels(s) OR labels(s) = ['Subtopic'])
        AND (type(r2) = 'PARENT_TOPIC' OR type(r2) = 'parent_topic')
        AND (labels(t) = ['TOPIC'] OR 'TOPIC' IN labels(t) OR labels(t) = ['Topic'])
        RETURN e, r1, s, r2, t
        """
        run_stage("Topic Hierarchy", topic_query, {})

    except Exception as exc:  # noqa: BLE001
        logger.error("Subgraph expansion failed: %s", exc)
        # We return whatever we found so far
        
    return nodes, edges, timings


def filter_subgraph_by_centrality(
    nodes: Dict[str, Any],
    edges: List[Dict[str, Any]],
    seed_entities: List[str],
    top_n: int = 10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Keep the entire connected component from seed entities.
    
    Strategy:
      1. Find all seed nodes by matching entity names.
      2. Expand to the full connected component (all nodes reachable from seeds).
      3. Apply a high safety cap (1000 nodes) only to prevent memory issues.
      4. Always include temporal nodes (Day/Segment) regardless of connectivity.
    
    This ensures the LLM sees the complete graph structure including temporal
    information without aggressive filtering that might drop Day/Segment nodes.
    """
    # Normalize seed entities for case-insensitive matching if they are strings
    seed_entities_lower = {s.lower() for s in seed_entities if isinstance(s, str)}
    seed_entities_set = set(seed_entities)

    # Find seed nodes by matching node IDs (primary) or entity names/titles (fallback)
    seed_node_ids: set[str] = set()
    for node_id, node_data in nodes.items():
        p_id = node_data.get("id") # The property-based ID (e.g. 'mom')
        props = node_data.get("properties", {})
        name = props.get("name")
        title = props.get("title")
        
        # Check ID match (internal integer ID or property ID)
        if str(node_id) in seed_entities_set or p_id in seed_entities_set:
            seed_node_ids.add(node_id)
            continue

        # Check Name/Title match (exact or case-insensitive)
        found = False
        for val in [p_id, name, title]:
            if val and isinstance(val, str):
                if val in seed_entities_set or val.lower() in seed_entities_lower:
                    seed_node_ids.add(node_id)
                    found = True
                    break
        if found:
            continue

    # If no seeds found, return empty
    if not seed_node_ids:
        return {}, []

    # Expand to full connected component from seeds
    # Expand to full connected component from seeds using BFS
    # This is O(N + E) instead of O(N * E) iterative relaxation
    
    # 1. Build Adjacency List
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        start_id = edge.get("start")
        end_id = edge.get("end")
        if start_id and end_id:
            if start_id not in adj:
                adj[start_id] = []
            if end_id not in adj:
                adj[end_id] = []
            adj[start_id].append(end_id)
            adj[end_id].append(start_id)  # Treat as undirected for connectivity

    # 2. BFS
    connected_nodes = set()
    queue = list(seed_node_ids)
    connected_nodes.update(seed_node_ids)
    
    # Optional: Safety cap for traversal depth/count if needed, but we rely on output cap later
    idx = 0
    while idx < len(queue):
        current = queue[idx]
        idx += 1
        
        if current in adj:
            for neighbor in adj[current]:
                if neighbor not in connected_nodes:
                    connected_nodes.add(neighbor)
                    queue.append(neighbor)

    # Merge connected component with temporal nodes
    # Only include temporal nodes that are actually connected to the seed context
    # This prevents pulling in irrelevant segments just because they exist in the graph
    selected_node_ids = connected_nodes
    
    # Optional: If we want to be safe, we can add temporal nodes that are 1 hop away from connected component?
    # For now, strict connectivity is better for relevance.
    
    # Apply a stricter safety cap to prevent context window overflow
    # Reduced from 1000 to 200 to prioritize high-signal nodes
    max_safety_cap = 500
    if len(selected_node_ids) > max_safety_cap:
        # If over cap, prioritize: seeds -> connected
        prioritized = seed_node_ids.copy()
        for node_id in connected_nodes:
            if len(prioritized) >= max_safety_cap:
                break
            prioritized.add(node_id)
        selected_node_ids = prioritized
        logger.warning(
            "Subgraph exceeded safety cap (%d nodes), truncated to %d nodes",
            len(connected_nodes),
            len(selected_node_ids),
        )

    path_nodes = selected_node_ids.copy()

    filtered_nodes = {
        node_id: node_data for node_id, node_data in nodes.items() if node_id in path_nodes
    }
    filtered_edges = [
        edge
        for edge in edges
        if edge.get("start") in path_nodes and edge.get("end") in path_nodes
    ]

    type_counts: Dict[str, int] = {}
    for node_data in filtered_nodes.values():
        props = node_data.get("properties", {})
        labels = node_data.get("labels", [])
        labels_lower = {label.lower() for label in labels} if labels else set()
        if "chunk" in labels_lower or props.get("text"):
            node_type = "Chunk"
        elif "segment" in labels_lower or props.get("content"):
            node_type = "Segment"
        elif "day" in labels_lower or props.get("date") and (props.get("episode_count") or props.get("segment_count")):
            node_type = "Day"
        else:
            node_type = "Entity"
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    logger.info(
        "Filtered subgraph: %s nodes (from %s), %s edges (from %s). Node types: %s",
        len(filtered_nodes),
        len(nodes),
        len(filtered_edges),
        len(edges),
        type_counts,
    )

    return filtered_nodes, filtered_edges


def format_graph_context(nodes: dict, edges: list) -> str:
    """
    Format graph data using XML structure for better model adherence.
    Prioritizes high-signal summaries and limits verbose text.
    """
    if not nodes:
        return "No relevant information found in the knowledge base."

    # Pre-process nodes into types for easier sectioning
    grouped = {
        "documents": [],
        "entities": [],
        "timeline": [],
        "topics": []
    }

    # Helper to clean names
    def get_name(n):
        p = n.get("properties", {})
        # Prioritize calculated ID which may contain the name from node.id
        return clean_entity_name(n.get("id") or p.get("name") or p.get("title") or "Unknown")

    for n in nodes.values():
        props = n.get("properties", {})
        labels = [l.lower() for l in n.get("labels", [])]
        
        # Categorize
        if "chunk" in labels or props.get("text"):
            grouped["documents"].append(n)
        elif "segment" in labels or "day" in labels or "conversation" in labels or props.get("date") or props.get("time"):
            grouped["timeline"].append(n)
        elif "topic" in labels or "subtopic" in labels:
            grouped["topics"].append(n)
        else:
            grouped["entities"].append(n)

    parts = []

    # 1. TOPICS (High-level context)
    if grouped["topics"]:
        parts.append("<topics>")
        for n in grouped["topics"]:
            p = n.get("properties", {})
            title = p.get("title", "Topic")
            summary = p.get("summary", "")
            if summary:
                parts.append(f'<topic name="{title}">{summary}</topic>')
        parts.append("</topics>\n")

    # 2. TIMELINE (Chronological context)
    if grouped["timeline"]:
        # Map segments to chunks to get knowledge triplets
        segment_triplets = {}
        for edge in edges:
            if edge.get("type") == "HAS_CHUNK":
                s_id, c_id = edge.get("start"), edge.get("end")
                if s_id in nodes and c_id in nodes:
                    chunk = nodes[c_id]
                    triplets = chunk.get("properties", {}).get("knowledge_triplets", [])
                    if triplets and isinstance(triplets, list):
                        segment_triplets.setdefault(s_id, []).extend(triplets)

        parts.append("<timeline>")
        # Sort by date
        sorted_timeline = sorted(
            grouped["timeline"], 
            key=lambda x: (x.get("properties") or {}).get("date") or (x.get("properties") or {}).get("time") or (x.get("properties") or {}).get("document_date") or ""
        )
        for n in sorted_timeline:
            p = n.get("properties", {})
            date = p.get("date") or p.get("time") or p.get("document_date")
            
            # Use knowledge triplets if available
            seg_id = n.get("element_id") or n.get("id")
            triplets = segment_triplets.get(seg_id, [])
            
            # Find connected Place/Context via edges
            associated_info = []
            related_chunks = []
            
            # Find chunks for this segment/conversation
            for edge in edges:
                 if edge.get("type") in ["HAS_CHUNK", "HAPPENED_AT", "HAS_CONTEXT"]:
                      # Check if this timeline node is the source
                      source_id = edge.get("start")
                      target_id = edge.get("end")
                      
                      # Handle potential ID mismatches (int vs str) by string comparison if needed, or rely on exact match
                      if str(source_id) == str(seg_id) and target_id in nodes:
                           target_node = nodes[target_id]
                           t_labels = target_node.get("labels", [])
                           
                           if "PLACE" in t_labels:
                                p_name = target_node.get("properties", {}).get("name", "Unknown Place")
                                associated_info.append(f"Location: {p_name}")
                           elif "CONTEXT" in t_labels:
                                desc = target_node.get("properties", {}).get("description", "")
                                if desc: associated_info.append(f"Context: {desc}")
                           elif "CHUNK" in t_labels:
                                related_chunks.append(target_node)

            body_parts = []
            if associated_info:
                 body_parts.extend(associated_info)

            if triplets:
                formatted_triplets = []
                for t in triplets:
                    if isinstance(t, list) and len(t) >= 3:
                        formatted_triplets.append(f"{t[0]} --[{t[1]}]--> {t[2]}")
                    elif isinstance(t, str):
                        formatted_triplets.append(t)
                
                unique_triplets = sorted(list(set(formatted_triplets)))
                if unique_triplets:
                    body_parts.append("Key Events/Relations:")
                    body_parts.extend(unique_triplets)
            
            # Add Chunk Text as Description (since strict source_documents is removed)
            # Only add if we have related chunks
            if related_chunks:
                 chunk_texts = []
                 for c in related_chunks:
                      c_text = c.get("properties", {}).get("text", "")
                      if c_text:
                           chunk_texts.append(c_text)
                 
                 if chunk_texts:
                      # Limit length if too long? For now, include relevant text.
                      # Ideally we summary, but here we need specific details like "notifications"
                      combined_text = "\n".join(chunk_texts)
                      body_parts.append(f"Description: {combined_text}")

            if body_parts:
                body = "\n".join(body_parts)
                parts.append(f'<event date="{date}">{body}</event>')
        parts.append("</timeline>\n")

    # 3. ENTITIES (Knowledge Graph nodes)
    if grouped["entities"]:
        parts.append("<entities>")
        # Sort by importance (PageRank) if available
        sorted_entities = sorted(
            grouped["entities"],
            key=lambda x: (x.get("properties") or {}).get("pagerank_centrality", 0),
            reverse=True
        )
        
        for n in sorted_entities[:10]: # Limit to top 10 entities
            p = n.get("properties", {})
            name = get_name(n)
            etype = p.get("entity_type", "Entity")
            summary = p.get("centrality_summary") or ""
            
            # Filter out generic structural boilerplate
            # The context is already sorted by PageRank, so structural importance is implied.
            boilerplate_markers = [
                "highly important in the network",
                "showing strong centrality",
                "centrality across multiple measures"
            ]
            if any(marker in summary for marker in boilerplate_markers):
                summary = ""
            
            # Compact format: Name (Type): Summary
            if summary:
                parts.append(f'<entity name="{name}" type="{etype}">{summary}</entity>')
            else:
                parts.append(f'<entity name="{name}" type="{etype}"/>')
        parts.append("</entities>\n")

    # 4. RELATIONSHIPS (Entity-Entity connection)
    # We filter out structural edges (HAS_ENTITY, HAS_CHUNK, etc)
    relevant_rels = []
    
    # Pre-fetch node names for fast lookup
    node_names = {}
    for nid, n in nodes.items():
        node_names[nid] = get_name(n)
        
    seen_rels = set()
    
    for edge in edges:
        rtype = edge.get("type") or edge.get("relation")
        if not rtype: 
            continue
            
        # Skip structural hierarchy edges
        if rtype in ["HAS_ENTITY", "HAS_CHUNK", "HAS_SEGMENT", "IN_TOPIC", "PARENT_TOPIC"]:
            continue
            
        s_id = edge.get("start") or edge.get("src_node")
        e_id = edge.get("end") or edge.get("dest_node")
        
        if s_id in nodes and e_id in nodes:
            s_name = node_names.get(s_id, "Unknown")
            e_name = node_names.get(e_id, "Unknown")
            
            # Key for deduplication
            rel_key = f"{s_name}-{rtype}-{e_name}"
            if rel_key not in seen_rels:
                relevant_rels.append(f'{s_name} --[{rtype}]--> {e_name}')
                seen_rels.add(rel_key)

    if relevant_rels:
        parts.append("<relationships>")
        # Limit to avoid token explosion, prioritize first 50 found
        for rel_str in relevant_rels[:50]:
            parts.append(rel_str)
        parts.append("</relationships>\n")

    # 5. SOURCE DOCUMENTS (Deep Dive) - REMOVED per user request
    # if grouped["documents"]:
    #     parts.append("<source_documents>")
    #     for i, n in enumerate(grouped["documents"], 1):
    #         p = n["properties"]
    #         text = p.get("text", "")
    #         # Skip metadata title as it often contains generative artifacts
    #         title = f"Document {i}"
    #         
    #         if text:
    #             parts.append(f'<document title="{title}">\n{text}\n</document>')
    #     parts.append("</source_documents>")

    return "\n".join(parts)


def merge_graph_data(
    accumulated: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge accumulated graph data with new graph data, avoiding duplicates.
    
    Args:
        accumulated: Existing graph data with 'nodes' and 'edges' keys
        new: New graph data with 'nodes' and 'edges' keys
        
    Returns:
        Merged graph data with unique nodes and edges
    """
    # Create a set of node IDs from accumulated graph to avoid duplicates
    accumulated_node_ids = {
        node.get("element_id") or node.get("id")
        for node in accumulated.get("nodes", [])
        if node.get("element_id") or node.get("id")
    }
    
    # Add new nodes that aren't already in accumulated
    merged_nodes = list(accumulated.get("nodes", []))
    for node in new.get("nodes", []):
        node_id = node.get("element_id") or node.get("id")
        if node_id and node_id not in accumulated_node_ids:
            merged_nodes.append(node)
            accumulated_node_ids.add(node_id)
    
    # Create a set of edge tuples (source, target, type) from accumulated graph
    accumulated_edges = {
        (
            edge.get("source") or edge.get("from"),
            edge.get("target") or edge.get("to"),
            edge.get("type") or edge.get("properties", {}).get("label", ""),
        )
        for edge in accumulated.get("edges", [])
    }
    
    # Add new edges that aren't already in accumulated
    merged_edges = list(accumulated.get("edges", []))
    for edge in new.get("edges", []):
        edge_tuple = (
            edge.get("source") or edge.get("from"),
            edge.get("target") or edge.get("to"),
            edge.get("type") or edge.get("properties", {}).get("label", ""),
        )
        if edge_tuple not in accumulated_edges:
            merged_edges.append(edge)
            accumulated_edges.add(edge_tuple)
    
    return {
        "nodes": merged_nodes,
        "edges": merged_edges,
    }



def enrich_with_triplets(nodes: Dict[str, Any], edges: List[Dict[str, Any]]):
    """
    Parse 'knowledge_triplets' from Chunk nodes and add implied entities/relationships to the graph data.
    """
    new_nodes = {}
    new_edges = []
    
    for nid, node in nodes.items():
        props = node.get("properties", {})
        triplets = props.get("knowledge_triplets")
        
        if triplets and isinstance(triplets, list):
             for triplet in triplets:
                 if len(triplet) != 3: continue
                 subj, pred, obj = triplet
                 
                 if not subj or not obj: continue

                 # Create/Find Subject Node
                 s_id = f"ent:{subj.strip()}"
                 if s_id not in nodes and s_id not in new_nodes:
                     new_nodes[s_id] = {
                         "id": s_id,
                         "labels": ["Entity", "FromTriplet"],
                         "properties": {"name": subj, "entity_type": "Entity", "generated": True}
                     }
                 
                 # Create/Find Object Node
                 o_id = f"ent:{obj.strip()}"
                 if o_id not in nodes and o_id not in new_nodes:
                     new_nodes[o_id] = {
                         "id": o_id,
                         "labels": ["Entity", "FromTriplet"],
                         "properties": {"name": obj, "entity_type": "Entity", "generated": True}
                     }
                 
                 # Add Relation Edge
                 edge = {
                     "type": pred.upper().replace(" ", "_"),
                     "start": s_id, 
                     "end": o_id,   
                     "properties": {"implied": True}
                 }
                 new_edges.append(edge)
                 
                 # Link Chunk to Subject (structural)
                 c_s_edge = {
                     "type": "MENTIONS",
                     "start": nid,
                     "end": s_id, 
                     "properties": {"implied": True}
                 }
                 new_edges.append(c_s_edge)
                 
                 # Link Chunk to Object (structural)
                 c_o_edge = {
                     "type": "MENTIONS",
                     "start": nid,
                     "end": o_id,
                     "properties": {"implied": True}
                 }
                 new_edges.append(c_o_edge)

    # Merge into main
    nodes.update(new_nodes)
    edges.extend(new_edges)


def retrieve_subgraph(
    db: GraphDB,
    keywords: List[str],
    query_text: str = "",
    accumulated_graph: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], List[str], Dict[str, float]]:
    """
    Retrieve relevant context using graph traversal.
    Optionally merges with accumulated graph data from previous queries.
    Returns:
        xml_context: str
        graph_data: dict
        seed_entities: list
        timings: dict
    """
    from src.app.infrastructure.llm import (  # noqa: WPS433
        get_embedding_model,
    )
    
    total_timings: Dict[str, float] = {}

    context_text = ""
    graph_data: Dict[str, Any] = {"nodes": [], "edges": []}
    seed_entities: List[str] = []

    try:
        query_embedding: List[float] = []
        if query_text:
            from src.app.infrastructure.llm import (  # noqa: WPS433
                get_embedding_model,
            )

            embedding_model = get_embedding_model()
            with Profiler("Query Embedding"):
                query_embedding = embedding_model.embed_query(query_text)

        with Profiler("Get Seed Entities"):
            seed_entities, seed_timings = get_seed_entities(db, query_embedding, keywords)
        total_timings.update(seed_timings)
        logger.info("Selected seed entities: %s", seed_entities)

        with Profiler("Expand Subgraph"):
            nodes, edges, expand_timings = expand_subgraph(db, seed_entities)
        total_timings.update(expand_timings)

        node_types_before: Dict[str, int] = {}
        for node_data in nodes.values():
            labels = node_data.get("labels", [])
            props = node_data.get("properties", {})
            labels_lower = {label.lower() for label in labels} if labels else set()
            if "chunk" in labels_lower or props.get("text"):
                node_type = "Chunk"
            elif "segment" in labels_lower or props.get("content"):
                node_type = "Segment"
            elif "subtopic" in labels_lower:
                node_type = "Subtopic"
            elif "topic" in labels_lower:
                node_type = "Topic"
            elif "day" in labels_lower or (props.get("date") and (props.get("episode_count") or props.get("segment_count"))):
                node_type = "Day"
            else:
                node_type = "Entity"
            node_types_before[node_type] = node_types_before.get(node_type, 0) + 1

        logger.info(
            "Subgraph expanded: %s nodes, %s edges. Node types before filtering: %s",
            len(nodes),
            len(edges),
            node_types_before,
        )

        # Enrich with implied entities from triplets (Fallback for missing edges)
        enrich_with_triplets(nodes, edges)

        # Keep entire connected component - no aggressive filtering
        # This ensures Day/Segment nodes are always included for temporal reasoning
        # This ensures Day/Segment nodes are always included for temporal reasoning
        with Profiler("Filter Subgraph"):
            nodes, edges = filter_subgraph_by_centrality(
                nodes, edges, seed_entities, top_n=1000
            )

        new_graph_data = {
            "nodes": list(nodes.values()),
            "edges": edges,
        }

        if accumulated_graph:
            graph_data = merge_graph_data(accumulated_graph, new_graph_data)
        else:
            graph_data = new_graph_data

        nodes_dict = {
            node.get("element_id") or node.get("id"): node for node in graph_data["nodes"]
        }
        with Profiler("Format Graph Context"):
            context_text = format_graph_context(nodes_dict, graph_data["edges"])

    except Exception as exc:  # noqa: BLE001
        logger.error("Graph retrieval failed: %s", exc)
        return f"Error retrieving graph data: {exc}", graph_data, [], total_timings

    return context_text, graph_data, seed_entities, total_timings


def run_focused_retrieval(
    db: GraphDB,
    query: str,
    messages: Optional[List[Message]] = None,
    accumulated_graph: Optional[Dict[str, Any]] = None,
) -> QueryResult:
    """Run focused retrieval pipeline with graph stats and memory tracking."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    start_time = time.time()
    query_memory = get_query_memory_usage()
    if messages is None:
        messages = []

    try:
        llm = get_llm()

        # Parallelize independent tasks: Keywords, Graph Stats, Memory
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Stats are independent
            future_stats = executor.submit(get_graph_stats, db)
            
            # Keywords depend on LLM but not DB
            context_query = query
            if messages:
                last_user_msg = next(
                    (m.content for m in reversed(messages) if m.role == "user"), None
                )
                if last_user_msg:
                    context_query = f"{last_user_msg} {query}"
            
            future_keywords = executor.submit(extract_keywords, llm, context_query)
            
            # Wait for keywords to proceed with retrieval
            try:
                keywords = future_keywords.result()
                msg = f"Keywords extracted: {keywords} (took {time.time() - t0:.2f}s)"
                logger.info(msg)
                print(msg, flush=True)
            except Exception as e: # noqa: BLE001
                logger.error("Keyword extraction failed in parallel: %s", e)
                keywords = []

        # Retrieve subgraph (depends on keywords)
            t1 = time.time()
            with Profiler("Total Subgraph Retrieval"):
                context, graph_data, seed_entities, retrieval_timings = retrieve_subgraph(
                    db, keywords, query, accumulated_graph
                )
            msg = f"Subgraph retrieval took {time.time() - t1:.2f}s"
            logger.info(msg)
            print(msg, flush=True)
            
            # Get stats result when needed
            try:
                graph_stats = future_stats.result()
            except Exception as e: # noqa: BLE001
                logger.warning("Graph stats failed in parallel: %s", e)
                graph_stats = {}

        system_prompt = """You are a Personal Life Assistant. You help users recall memories, understand their daily patterns, and answer questions about their life based on their logs.

You have access to a knowledge graph structured in XML tags:
- <topics>: High-level themes of the user's life.
- <timeline>: Chronological events (Days, Segments, Conversations).
- <entities>: Key people, places, and concepts.
- <relationships>: Connections between entities.
- <source_documents>: Full text transcripts and descriptions.

**Guidelines:**
- Answer differently based on the user's question type (memory recall, pattern analysis, etc.).
- BE PERSONAL. Use "you" and "your".
- Cite dates and times specifically.
- Use the <timeline> to order events chronologically.
- If asking about a specific day, summarize the flow of that day.
- If information is missing, say so gently.

**Available Information:**
{context}
"""

        lm_messages = [SystemMessage(content=system_prompt.format(context=context))]
        for msg in messages:
            if msg.role == "user":
                lm_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lm_messages.append(AIMessage(content=msg.content))
        lm_messages.append(HumanMessage(content=query))

        # Capture full prompt for debug
        full_prompt_debug = "=== SYSTEM PROMPT ===\n" + system_prompt.format(context=context) + "\n\n"
        for msg in messages:
             full_prompt_debug += f"=== {msg.role.upper()} ===\n{msg.content}\n\n"
        full_prompt_debug += f"=== USER (CURRENT) ===\n{query}"

        t2 = time.time()
        logger.info("Starting LLM generation with context length %d chars...", len(context))
        with Profiler("LLM Answer Generation"):
             response = llm.invoke(lm_messages)
        answer = response.content
        logger.info("LLM generation took %.2fs", time.time() - t2)

        execution_time = time.time() - start_time
        final_memory = get_query_memory_usage()
        query_memory["peak_mb"] = final_memory["before_mb"]
        query_memory["delta_mb"] = round(
            final_memory["before_mb"] - query_memory["before_mb"], 1
        )

        return QueryResult(
            answer=answer,
            context=context,
            execution_time=execution_time,
            keywords=keywords,
            graph_data=graph_data,
            seed_entities=seed_entities,
            graph_stats=graph_stats,
            query_memory_mb=query_memory,
            full_prompt=full_prompt_debug,
            detailed_timing=retrieval_timings
        )

    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline failed: %s", e)
        execution_time = time.time() - start_time
        final_memory = get_query_memory_usage()
        query_memory["peak_mb"] = final_memory["before_mb"]
        query_memory["delta_mb"] = round(
            final_memory["before_mb"] - query_memory["before_mb"], 1
        )

        return QueryResult(
            answer=f"I encountered an error: {str(e)}",
            context="",
            execution_time=execution_time,
            keywords=[],
            graph_data={},
            seed_entities=[],
            graph_stats=get_graph_stats(db),
            query_memory_mb=query_memory,
        )
