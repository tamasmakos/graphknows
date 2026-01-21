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
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import resource
import networkx as nx

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
    seed_topics: List[str] = []
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
) -> Tuple[List[str], List[str], Dict[str, float]]:
    """
    Identify and rerank seed entities/topics.
    Returns:
        List of entity seeds
        List of topic seeds
        Dictionary of timing stats
    """
    candidates: Dict[str, float] = {}  # name -> score
    topic_candidates: Dict[str, float] = {}
    timings: Dict[str, float] = {}

    def search_topic():
        t0 = time.time()
        local_candidates = {}
        local_topics = {}
        if not query_embedding:
            return local_candidates, local_topics, 0.0
        try:
            topic_results = db.query_vector(
                "TOPIC", query_embedding, k=3, min_score=0.55
            )
            for node_data, vector_score in topic_results:
                topic_id = node_data.get("id")
                if topic_id:
                     local_topics[topic_id] = max(local_topics.get(topic_id, 0), vector_score)
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
                         combined_score = vector_score * 0.5 + pr * 0.5
                         local_candidates[eid] = max(local_candidates.get(eid, 0), combined_score)
        except Exception as e:
            logger.warning("Topic search failed: %s", e)
        return local_candidates, local_topics, time.time() - t0

    def search_subtopic():
        t0 = time.time()
        local_candidates = {}
        local_topics = {}
        if not query_embedding:
            return local_candidates, local_topics, 0.0
        try:
            subtopic_results = db.query_vector(
                "SUBTOPIC", query_embedding, k=3, min_score=0.55
            )
            for node_data, vector_score in subtopic_results:
                subtopic_id = node_data.get("id")
                if subtopic_id:
                     local_topics[subtopic_id] = max(local_topics.get(subtopic_id, 0), vector_score)
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
        except Exception as e:
            logger.warning("Subtopic search failed: %s", e)
        return local_candidates, local_topics, time.time() - t0

    def search_entity():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            entity_results = db.query_vector(
                "ENTITY_CONCEPT", query_embedding, k=10, min_score=0.5
            )
            for node_data, score in entity_results:
                entity_id = node_data.get("id")
                if entity_id:
                    local_candidates[entity_id] = max(local_candidates.get(entity_id, 0), score)
        except Exception as e:
            logger.warning("Entity vector search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_keywords():
        t0 = time.time()
        local_ents = {}
        local_topics = {}
        if not keywords:
            return local_ents, local_topics, 0.0
        try:
            keyword_query = """
            UNWIND $keywords AS keyword
            MATCH (seed)
            WHERE (seed:ENTITY_CONCEPT OR seed:TOPIC OR seed:SUBTOPIC OR seed:DAY OR seed:CONVERSATION OR seed:PLACE)
            AND (
                toLower(coalesce(seed.name, "")) CONTAINS toLower(keyword)
                OR toLower(coalesce(seed.id, "")) CONTAINS toLower(keyword)
                OR toLower(coalesce(seed.title, "")) CONTAINS toLower(keyword)
            )
            RETURN DISTINCT seed.id AS id, labels(seed) as labels
            LIMIT 100
            """
            results = db.query(keyword_query, {"keywords": keywords})
            for record in results:
                eid = record.get("id")
                labels = [l.upper() for l in record.get("labels", [])]
                if "TOPIC" in labels or "SUBTOPIC" in labels:
                    local_topics[eid] = max(local_topics.get(eid, 0), 1.0)
                else:
                    local_ents[eid] = max(local_ents.get(eid, 0), 1.0)
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
        return local_ents, local_topics, time.time() - t0

    with Profiler("Parallel Seed Search"), ThreadPoolExecutor(max_workers=4) as executor:
        f_topic = executor.submit(search_topic)
        f_sub = executor.submit(search_subtopic)
        f_ent = executor.submit(search_entity)
        f_key = executor.submit(search_keywords)

        r_topic, t_topic, d_topic = f_topic.result()
        r_sub, t_sub, d_sub = f_sub.result()
        r_ent, d_ent = f_ent.result()
        r_key_ent, r_key_top, d_key = f_key.result()

        timings.update({"vector_topic": d_topic, "vector_sub": d_sub, "vector_ent": d_ent, "keyword": d_key})
        
        for d in [r_topic, r_sub, r_ent, r_key_ent]:
            for k, v in d.items(): candidates[k] = max(candidates.get(k, 0), v)
        for d in [t_topic, t_sub, r_key_top]:
            for k, v in d.items(): topic_candidates[k] = max(topic_candidates.get(k, 0), v)

    if not candidates and not topic_candidates:
        return [], [], timings

    # Rerank entities
    final_entities = []
    
    if candidates:
        try:
            rerank_query = "UNWIND $ids AS id MATCH (e) WHERE e.id = id RETURN e.id AS id, coalesce(e.pagerank_centrality, 0.0) AS pagerank"
            results = db.query(rerank_query, {"ids": list(candidates.keys())})
            scored = []
            for r in results:
                name, pr = r.get("id"), r.get("pagerank", 0.0)
                scored.append((name, candidates.get(name, 0)*0.7 + pr*0.3))
            scored.sort(key=lambda x: x[1], reverse=True)
            final_entities = [x[0] for x in scored[:30]]
        except Exception:
            final_entities = list(candidates.keys())[:20]

    # Rerank topics
    scored_topics = sorted(topic_candidates.items(), key=lambda x: x[1], reverse=True)
    final_topics = [x[0] for x in scored_topics[:10]]

    return final_entities, final_topics, timings


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
    internal_to_string_id = {}
    
    # 1. First pass: Process all nodes to establish ID mappings
    for record in results:
        for key, value in record.items():
            candidate_node = None
            if hasattr(value, "labels") and hasattr(value, "id"): # Object
                candidate_node = value
            elif isinstance(value, dict) and ("labels" in value or "properties" in value): # Dict
                candidate_node = value
            
            if candidate_node:
                node_props: Dict[str, Any] = {}
                labels: List[str] = []
                internal_id = ""

                if hasattr(candidate_node, "properties"):
                    raw_props = candidate_node.properties
                    node_props = dict(raw_props) if hasattr(raw_props, "items") else {}
                    
                    labels = list(candidate_node.labels) if hasattr(candidate_node, "labels") else []
                    internal_id = str(candidate_node.id)
                elif isinstance(candidate_node, dict):
                    node_props = candidate_node.get("properties", {}).copy()
                    labels = list(candidate_node.get("labels", []))
                    internal_id = str(candidate_node.get("id") or candidate_node.get("element_id") or "")

                # Establish the "canonical" ID for this node
                # Preference: 'id' property > 'p_id' property > internal ID
                string_id = node_props.get('id') or node_props.get('p_id') or internal_id
                internal_to_string_id[internal_id] = string_id
                
                if string_id not in nodes:
                    filtered_props = filter_node_properties(node_props, labels)
                    nodes[string_id] = {
                        "id": string_id,
                        "element_id": internal_id,
                        "display_id": filtered_props.get("name") or filtered_props.get("title") or string_id,
                        "labels": labels,
                        "properties": filtered_props,
                    }

    # 2. Second pass: Process all relationships using established ID mapping
    for record in results:
        for key, value in record.items():
            candidate_rel = None
            if hasattr(value, "src_node") and hasattr(value, "dest_node"): # FalkorDB Object
                candidate_rel = value
            elif isinstance(value, dict) and "start" in value and "end" in value: # Dict
                candidate_rel = value
            
            if candidate_rel:
                rel_props = {}
                rel_type = "RELATED"
                src_internal = ""
                dst_internal = ""

                if hasattr(candidate_rel, "properties"):
                    rel_props = dict(candidate_rel.properties)
                    rel_type = getattr(candidate_rel, "relation", getattr(candidate_rel, "type", "RELATED"))
                    src_internal = str(candidate_rel.src_node)
                    dst_internal = str(candidate_rel.dest_node)
                elif isinstance(candidate_rel, dict):
                    rel_props = candidate_rel.get("properties", {}).copy()
                    rel_type = candidate_rel.get("type", "RELATED")
                    src_internal = str(candidate_rel.get("start", ""))
                    dst_internal = str(candidate_rel.get("end", ""))

                # Map internal IDs to our canonical string IDs
                src_id = internal_to_string_id.get(src_internal, src_internal)
                dst_id = internal_to_string_id.get(dst_internal, dst_internal)

                if src_id and dst_id:
                    edges.append({
                        "source": src_id,
                        "target": dst_id,
                        "start": src_id,
                        "end": dst_id,
                        "type": rel_type,
                        "properties": rel_props
                    })

def expand_subgraph(
    db: GraphDB, seed_entities: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    """
    Expand subgraph using a dynamic BFS traversal from seed entities.
    """
    nodes: Dict[str, Any] = {}
    edges: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}

    if not seed_entities:
        return nodes, edges, timings

    max_hops = 5
    nodes_per_hop_limit = 500
    
    current_frontier = set([str(s).strip() for s in seed_entities if s])
    visited_ids = set(current_frontier)
    
    def check_limits():
        return len(nodes) >= MAX_EXPANDED_NODES

    try:
        t0 = time.time()
        with Profiler("Fetch Seed Nodes"):
            seed_list = list(current_frontier)
            if seed_list:
                query = """
                MATCH (n)
                WHERE n.id IN $ids 
                OR n.name IN $ids 
                OR n.title IN $ids
                RETURN n
                """
                result = db.query(query, {'ids': seed_list})
                _process_graph_results(result, nodes, edges)
                
                # Update frontier with actual canonical IDs found
                found_ids = set(nodes.keys())
                current_frontier = found_ids
                visited_ids.update(current_frontier)
                
        timings["fetch_seeds"] = time.time() - t0
        logger.info(f"BFS Start: Found {len(nodes)} seed nodes. Frontier size: {len(current_frontier)}")

        for hop in range(1, max_hops + 1):
            if check_limits() or not current_frontier:
                break
                
            t_hop = time.time()
            next_frontier = set()
            frontier_list = list(current_frontier)
            total_new_nodes = 0
            
            for i in range(0, len(frontier_list), 100):
                if check_limits(): break
                batch = frontier_list[i : i + 100]
                
                query = """
                MATCH (n)-[r]-(m)
                WHERE n.id IN $ids
                RETURN n, r, m
                LIMIT $limit
                """
                result = db.query(query, {'ids': batch, 'limit': nodes_per_hop_limit})
                
                nodes_before = len(nodes)
                _process_graph_results(result, nodes, edges)
                nodes_after = len(nodes)
                total_new_nodes += (nodes_after - nodes_before)
                
                # Collect new frontier from processed canonical IDs
                # All keys currently in 'nodes' that haven't been visited
                for nid in nodes.keys():
                    if nid not in visited_ids:
                        next_frontier.add(nid)
                        visited_ids.add(nid)

            timings[f"bfs_hop_{hop}"] = time.time() - t_hop
            logger.info(f"BFS Hop {hop}: Expanded {total_new_nodes} new nodes. Next Frontier: {len(next_frontier)}")
            current_frontier = next_frontier

    except Exception as exc:
        logger.error("BFS expansion failed: %s", exc)
        
    return nodes, edges, timings


def filter_subgraph_by_centrality(
    nodes: Dict[str, Any],
    edges: List[Dict[str, Any]],
    seed_entities: List[str],
    max_nodes: int = 70,  # Reduced default from 500/1000
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Smarter subgraph filtering:
    1. Identifies seed nodes.
    2. Calculates relevance scores based on distance from seeds and static PageRank.
    3. Prioritizes keeping the hierarchy (Day -> Episode -> Conv) for temporal queries.
    4. Limits total nodes to ensure a dense, high-signal context.
    """
    if not nodes:
        return {}, []

    seed_entities_set = set(str(s) for s in seed_entities)
    seed_node_ids: set[str] = set()

    # 1. Identify Seed Nodes
    for node_id, node_data in nodes.items():
        if node_id in seed_entities_set:
            seed_node_ids.add(node_id)
            continue
        
        p_id = str(node_data.get("properties", {}).get("id", ""))
        display_id = str(node_data.get("display_id", ""))
        if p_id in seed_entities_set or display_id in seed_entities_set:
            seed_node_ids.add(node_id)

    # Fallback: if no seeds match, return a limited set of important nodes
    if not seed_node_ids:
        logger.warning(f"filter_subgraph: No seed node IDs matched. seeds={seed_entities_set}")
        # Just return top 30 nodes by PageRank if no seeds
        sorted_by_pr = sorted(
            nodes.items(),
            key=lambda x: x[1].get("properties", {}).get("pagerank_centrality", 0),
            reverse=True
        )[:30]
        nodes_pr = {nid: nd for nid, nd in sorted_by_pr}
        return nodes_pr, []

    # 2. Build Adjacency for distance calculation
    G = nx.Graph()
    for nid, nd in nodes.items():
        G.add_node(nid)
    for edge in edges:
        s, t = edge.get("source"), edge.get("target")
        if s in nodes and t in nodes:
            G.add_edge(s, t)

    # 3. Calculate Relevance Scores
    # Score = (1 / (distance + 1)) * (1 + PageRank)
    relevance_scores = {}
    for node_id in nodes:
        # Calculate min distance to any seed
        min_dist = float('inf')
        for seed in seed_node_ids:
            try:
                # Limit search depth for speed
                dist = nx.shortest_path_length(G, source=node_id, target=seed)
                min_dist = min(min_dist, dist)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        if min_dist == float('inf'):
            # Check if it's a temporal node (important for context anyway)
            labels = [l.lower() for l in nodes[node_id].get("labels", [])]
            if any(l in labels for l in ["day", "episode", "conversation"]):
                dist_factor = 0.2 # Give temporal nodes a chance
            else:
                dist_factor = 0.05
        else:
            dist_factor = 1.0 / (min_dist + 1)

        # PR Factor
        pr = nodes[node_id].get("properties", {}).get("pagerank_centrality", 0.01)
        
        # Node Type Boosting
        type_boost = 1.0
        labels_lower = [l.lower() for l in nodes[node_id].get("labels", [])]
        if "topic" in labels_lower or "subtopic" in labels_lower:
            type_boost = 2.0
        elif "day" in labels_lower:
            type_boost = 1.5
        elif "chunk" in labels_lower:
            type_boost = 0.8 # De-prioritize raw chunks if they are far

        relevance_scores[node_id] = dist_factor * (1 + pr) * type_boost

    # 4. Select Top Nodes
    # Ensure seeds are always kept
    top_node_ids = set(seed_node_ids)
    
    # Sort remaining nodes by score
    remaining_nodes = [nid for nid in nodes if nid not in seed_node_ids]
    remaining_nodes.sort(key=lambda nid: relevance_scores.get(nid, 0), reverse=True)
    
    # Fill up to max_nodes
    num_to_add = max(0, max_nodes - len(top_node_ids))
    top_node_ids.update(remaining_nodes[:num_to_add])

    # 5. Filter nodes and edges
    filtered_nodes = {nid: nd for nid, nd in nodes.items() if nid in top_node_ids}
    filtered_edges = [
        e for e in edges 
        if e.get("source") in top_node_ids and e.get("target") in top_node_ids
    ]

    # Final cleanup: ensure edges only link to existing nodes (redundant but safe)
    filtered_edges = [e for e in filtered_edges if e["source"] in filtered_nodes and e["target"] in filtered_nodes]

    logger.info(
        "Smarter Filter: Kept %d nodes (from %d), %d edges (from %d)",
        len(filtered_nodes), len(nodes), len(filtered_edges), len(edges)
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
        return clean_entity_name(n.get("display_id") or n.get("id") or p.get("name") or p.get("title") or "Unknown")

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
            
            # Add image_description if present directly on the node (new simplified schema)
            img_desc = p.get("image_description")
            if img_desc:
                associated_info.append(f"Visual Context: {img_desc}")
            
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
                     "source": s_id,
                     "target": o_id,
                     "start": s_id, 
                     "end": o_id,   
                     "properties": {"implied": True}
                 }
                 new_edges.append(edge)
                 
                 # Link Chunk to Subject (structural)
                 c_s_edge = {
                     "type": "MENTIONS",
                     "source": nid,
                     "target": s_id,
                     "start": nid,
                     "end": s_id, 
                     "properties": {"implied": True}
                 }
                 new_edges.append(c_s_edge)
                 
                 # Link Chunk to Object (structural)
                 c_o_edge = {
                     "type": "MENTIONS",
                     "source": nid,
                     "target": o_id,
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
) -> Tuple[str, Dict[str, Any], List[str], List[str], Dict[str, float]]:
    """
    Retrieve relevant context using graph traversal.
    Returns:
        xml_context: str
        graph_data: dict
        seed_entities: list
        seed_topics: list
        timings: dict
    """
    total_timings: Dict[str, float] = {}
    context_text = ""
    graph_data: Dict[str, Any] = {"nodes": [], "edges": []}
    seed_entities: List[str] = []
    seed_topics: List[str] = []

    try:
        query_embedding: List[float] = []
        if query_text:
            from src.app.infrastructure.llm import get_embedding_model
            embedding_model = get_embedding_model()
            with Profiler("Query Embedding"):
                query_embedding = embedding_model.embed_query(query_text)

        with Profiler("Get Seed Entities"):
            seed_entities, seed_topics, seed_timings = get_seed_entities(db, query_embedding, keywords)
        total_timings.update(seed_timings)
        
        all_seeds = list(set(seed_entities + seed_topics))

        with Profiler("Expand Subgraph"):
            nodes, edges, expand_timings = expand_subgraph(db, all_seeds)
        total_timings.update(expand_timings)

        # Smart Filter
        with Profiler("Filter Subgraph"):
            nodes, edges = filter_subgraph_by_centrality(nodes, edges, all_seeds, max_nodes=70)

        new_graph_data = {"nodes": list(nodes.values()), "edges": edges}
        if accumulated_graph:
            graph_data = merge_graph_data(accumulated_graph, new_graph_data)
        else:
            graph_data = new_graph_data

        nodes_dict = {node.get("id"): node for node in graph_data["nodes"]}
        with Profiler("Format Graph Context"):
            context_text = format_graph_context(nodes_dict, graph_data["edges"])

    except Exception as exc:
        logger.error("Graph retrieval failed: %s", exc)
        return f"Error: {exc}", graph_data, [], [], total_timings

    return context_text, graph_data, seed_entities, seed_topics, total_timings


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
                context, graph_data, seed_entities, seed_topics, retrieval_timings = retrieve_subgraph(
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
            seed_topics=seed_topics,
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
