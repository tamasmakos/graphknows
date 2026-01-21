"""
Graph retrieval utilities for seed entity discovery and subgraph expansion.

Extracted from retrieval.py to provide focused graph traversal logic.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from src.app.infrastructure.graph_db import GraphDB
from src.app.services.context_builder import filter_node_properties

logger = logging.getLogger(__name__)

# Limits to prevent graph explosion
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


def get_seed_entities(
    db: GraphDB, query_embedding: List[float], keywords: List[str]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Identify and rerank seed entities using vector search and keyword matching.
    Optimized to run searches in parallel.
    
    Returns:
        List of seed entity IDs
        Dictionary of timing stats
    """
    candidates: Dict[str, float] = {}
    timings: Dict[str, float] = {}

    def search_topic():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            topic_results = db.query_vector("TOPIC", query_embedding, k=3, min_score=0.55)
            for node_data, vector_score in topic_results:
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
                         combined_score = vector_score * 0.5 + pr * 0.5
                         local_candidates[eid] = max(local_candidates.get(eid, 0), combined_score)
        except Exception as e:
            logger.warning("Topic search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_subtopic():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            subtopic_results = db.query_vector("SUBTOPIC", query_embedding, k=3, min_score=0.55)
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
        except Exception as e:
            logger.warning("Subtopic search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_entity():
        t0 = time.time()
        local_candidates = {}
        if not query_embedding:
            return local_candidates, 0.0
        try:
            entity_results = db.query_vector("ENTITY_CONCEPT", query_embedding, k=10, min_score=0.5)
            for node_data, score in entity_results:
                entity_id = node_data.get("id")
                if entity_id:
                    local_candidates[entity_id] = max(local_candidates.get(entity_id, 0), score)
        except Exception as e:
            logger.warning("Entity vector search failed: %s", e)
        return local_candidates, time.time() - t0

    def search_keywords():
        t0 = time.time()
        local_candidates = {}
        if not keywords:
            return local_candidates, 0.0
        try:
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
        except Exception as e:
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
            except Exception as e:
                logger.error("Search %s failed: %s", key, e)
                timings[key] = 0.0

    if not candidates:
        return [], timings

    # Rerank by PageRank
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

    except Exception as e:
        logger.error("Reranking failed: %s", e)
        final_seeds = list(candidates.keys())[:5]
    timings["seed_reranking"] = time.time() - rerank_t0

    return final_seeds, timings


def _process_graph_results(results: List[Dict[str, Any]], nodes: Dict[str, Any], edges: List[Dict[str, Any]]):
    """
    Helper to process query results and update nodes/edges collections.
    Handles FalkorDB result formats and property extraction.
    """
    for record in results:
        for key, value in record.items():
            candidate_node = None
            if hasattr(value, "labels") and hasattr(value, "id"):
                candidate_node = value
            elif isinstance(value, dict) and ("labels" in value or "properties" in value):
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
                
                for k, v in list(node_props.items()):
                    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                        try:
                            node_props[k] = json.loads(v)
                        except Exception:
                            pass

                filtered_props = filter_node_properties(node_props, labels)
                
                node_display_id = (
                    filtered_props.get("name")
                    or filtered_props.get("title")
                    or node_props.get("id")
                    or element_id
                )

                nodes[element_id] = {
                    "id": element_id,  # Use internal ID for linking
                    "element_id": element_id,
                    "display_id": node_display_id, # Keep display ID if needed
                    "labels": labels,
                    "properties": filtered_props,
                }

        # Process Relationships
        for key, value in record.items():
            candidate_rel = None
            if hasattr(value, "start_node") and hasattr(value, "end_node"):
                candidate_rel = value
            elif hasattr(value, "src_node") and hasattr(value, "dest_node"):
                candidate_rel = value
            elif isinstance(value, dict) and "start" in value and "end" in value and "type" in value:
                candidate_rel = value
            
            if candidate_rel:
                rel_props: Dict[str, Any] = {}
                rel_type = "RELATED"
                start_id = ""
                end_id = ""

                if hasattr(candidate_rel, "properties"):
                    raw_props = candidate_rel.properties
                    rel_props = dict(raw_props) if hasattr(raw_props, "items") else {}
                    if hasattr(candidate_rel, "relation"):
                        rel_type = candidate_rel.relation
                    else:
                        rel_type = getattr(candidate_rel, "type", "RELATED")
                    
                    if hasattr(candidate_rel, "start_node"):
                        start_id = str(getattr(candidate_rel.start_node, "id", "")) or str(getattr(candidate_rel.start_node, "element_id", ""))
                        end_id = str(getattr(candidate_rel.end_node, "id", "")) or str(getattr(candidate_rel.end_node, "element_id", ""))
                    elif hasattr(candidate_rel, "src_node"):
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
                    edges.append({
                        "source": start_id,
                        "target": end_id,
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
    5. Fetch Topic Hierarchy
    6. Fetch Semantic Expansion
    
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
        
        seed_ids = [int(n_id) for n_id in nodes.keys() if n_id.isdigit()]
        if not seed_ids:
            seed_ids = list(nodes.keys())
        
        if not seed_ids:
            return nodes, edges, timings

        # 1.5 Expand Topic Content (New Step)
        # If we have Topic/Subtopic seeds, we must find the entities they contain to find chunks
        has_topics = any(
            "TOPIC" in n.get("labels", []) or "SUBTOPIC" in n.get("labels", [])
            or "Topic" in n.get("labels", []) or "Subtopic" in n.get("labels", [])
            for n in nodes.values()
        )
        
        if has_topics:
            logger.info("Topic/Subtopic seeds detected. Expanding to contained entities...")
            topic_expand_query = """
            MATCH (seed)
            WHERE ID(seed) IN $seed_ids
            OPTIONAL MATCH (seed)<-[:PARENT_TOPIC]-(sub:SUBTOPIC)<-[:IN_TOPIC]-(e1)
            WHERE ("Entity" IN labels(e1) OR "ENTITY_CONCEPT" IN labels(e1))
            OPTIONAL MATCH (seed)<-[:IN_TOPIC]-(e2)
            WHERE ("Entity" IN labels(e2) OR "ENTITY_CONCEPT" IN labels(e2))
            RETURN e1, e2
            LIMIT 100
            """
            run_stage("Topic Content", topic_expand_query, {"seed_ids": seed_ids})
            
            # Refresh seed_ids to include the newly found entities
            # This allows the subsequent "Connected Chunks" step to find chunks for these entities
            seed_ids = [int(n_id) for n_id in nodes.keys() if n_id.isdigit()]

        logger.info("Found %d seed nodes (including topic expansion). expanding chunks...", len(seed_ids))

        # 2. Fetch Connected Chunks
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
            batch_size = 100
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                batch_str = "[" + ", ".join(str(cid) for cid in batch) + "]"
                
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

            # 4. Fetch Context Neighbors
            seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                batch_str = "[" + ", ".join(str(cid) for cid in batch) + "]"
                
                neighbor_query = f"""
                MATCH (c)-[r:HAS_ENTITY]->(e)
                WHERE ID(c) IN {batch_str}
                AND NOT ID(e) IN {seed_ids_str}
                RETURN c, r, e
                ORDER BY coalesce(e.pagerank_centrality, 0.0) DESC
                LIMIT 200
                """
                run_stage(f"Context Neighbors Batch {i//batch_size}", neighbor_query, {})

        # 5. Topic Hierarchy
        logger.info("Expanding topic hierarchy for seeds...")
        seed_ids_str = "[" + ", ".join(str(sid) for sid in seed_ids) + "]"
        
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
        
        # 6. Semantic Expansion
        logger.info(f"Expanding semantic relationships for {len(seed_ids)} seeds...")
        
        semantic_query = f"""
        MATCH (s)-[r]-(e)
        WHERE ID(s) IN {seed_ids_str}
        AND (labels(e) = ['Entity'] OR 'Entity' IN labels(e) OR labels(e) = ['ENTITY_CONCEPT'] OR 'ENTITY_CONCEPT' IN labels(e))
        AND type(r) <> 'HAS_ENTITY'
        RETURN s, r, e
        ORDER BY coalesce(e.pagerank_centrality, 0.0) DESC
        LIMIT 200
        """
        run_stage("Semantic Expansion", semantic_query, {})

    except Exception as exc:
        logger.error("Subgraph expansion failed: %s", exc)
        
    return nodes, edges, timings


def filter_subgraph_by_centrality(
    nodes: Dict[str, Any],
    edges: List[Dict[str, Any]],
    seed_entities: List[str],
    max_nodes: int = 500,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Keep the entire connected component from seed entities.
    
    Uses BFS for O(N + E) traversal instead of iterative relaxation.
    """
    seed_entities_lower = {s.lower() for s in seed_entities if isinstance(s, str)}
    seed_entities_set = set(seed_entities)

    seed_node_ids: set[str] = set()
    for node_id, node_data in nodes.items():
        p_id = node_data.get("id")
        props = node_data.get("properties", {})
        name = props.get("name")
        title = props.get("title")
        
        if str(node_id) in seed_entities_set or p_id in seed_entities_set:
            seed_node_ids.add(node_id)
            continue

        for val in [p_id, name, title]:
            if val and isinstance(val, str):
                if val in seed_entities_set or val.lower() in seed_entities_lower:
                    seed_node_ids.add(node_id)
                    break

    if not seed_node_ids:
        return {}, []

    # Build Adjacency List
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        start_id = edge.get("start")
        end_id = edge.get("end")
        if start_id and end_id:
            adj.setdefault(start_id, []).append(end_id)
            adj.setdefault(end_id, []).append(start_id)

    # BFS
    connected_nodes = set(seed_node_ids)
    queue = list(seed_node_ids)
    idx = 0
    while idx < len(queue):
        current = queue[idx]
        idx += 1
        for neighbor in adj.get(current, []):
            if neighbor not in connected_nodes:
                connected_nodes.add(neighbor)
                queue.append(neighbor)

    selected_node_ids = connected_nodes
    
    if len(selected_node_ids) > max_nodes:
        prioritized = seed_node_ids.copy()
        for node_id in connected_nodes:
            if len(prioritized) >= max_nodes:
                break
            prioritized.add(node_id)
        selected_node_ids = prioritized
        logger.warning(
            "Subgraph exceeded safety cap (%d nodes), truncated to %d nodes",
            len(connected_nodes),
            len(selected_node_ids),
        )

    filtered_nodes = {
        node_id: node_data for node_id, node_data in nodes.items() if node_id in selected_node_ids
    }
    filtered_edges = [
        edge for edge in edges
        if edge.get("start") in selected_node_ids and edge.get("end") in selected_node_ids
    ]

    logger.info(
        "Filtered subgraph: %s nodes (from %s), %s edges (from %s)",
        len(filtered_nodes), len(nodes), len(filtered_edges), len(edges),
    )

    return filtered_nodes, filtered_edges


def enrich_with_triplets(nodes: Dict[str, Any], edges: List[Dict[str, Any]]):
    """
    Parse 'knowledge_triplets' from Chunk nodes and add implied entities/relationships.
    """
    new_nodes = {}
    new_edges = []
    
    for nid, node in nodes.items():
        props = node.get("properties", {})
        triplets = props.get("knowledge_triplets")
        
        if triplets and isinstance(triplets, list):
            for triplet in triplets:
                if len(triplet) != 3:
                    continue
                subj, pred, obj = triplet
                
                if not subj or not obj:
                    continue

                s_id = f"ent:{subj.strip()}"
                if s_id not in nodes and s_id not in new_nodes:
                    new_nodes[s_id] = {
                        "id": s_id,
                        "labels": ["Entity", "FromTriplet"],
                        "properties": {"name": subj, "entity_type": "Entity", "generated": True}
                    }
                
                o_id = f"ent:{obj.strip()}"
                if o_id not in nodes and o_id not in new_nodes:
                    new_nodes[o_id] = {
                        "id": o_id,
                        "labels": ["Entity", "FromTriplet"],
                        "properties": {"name": obj, "entity_type": "Entity", "generated": True}
                    }
                
                edge = {
                    "type": pred.upper().replace(" ", "_"),
                    "source": s_id,
                    "target": o_id,
                    "start": s_id, 
                    "end": o_id,   
                    "properties": {"implied": True}
                }
                new_edges.append(edge)
                
                new_edges.append({
                    "type": "MENTIONS",
                    "source": nid,
                    "target": s_id,
                    "start": nid,
                    "end": s_id, 
                    "properties": {"implied": True}
                })
                new_edges.append({
                    "type": "MENTIONS",
                    "source": nid,
                    "target": o_id,
                    "start": nid,
                    "end": o_id,
                    "properties": {"implied": True}
                })

    nodes.update(new_nodes)
    edges.extend(new_edges)


def merge_graph_data(
    accumulated: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge accumulated graph data with new graph data, avoiding duplicates."""
    accumulated_node_ids = {
        node.get("element_id") or node.get("id")
        for node in accumulated.get("nodes", [])
        if node.get("element_id") or node.get("id")
    }
    
    merged_nodes = list(accumulated.get("nodes", []))
    for node in new.get("nodes", []):
        node_id = node.get("element_id") or node.get("id")
        if node_id and node_id not in accumulated_node_ids:
            merged_nodes.append(node)
            accumulated_node_ids.add(node_id)
    
    accumulated_edges = {
        (
            edge.get("source") or edge.get("from"),
            edge.get("target") or edge.get("to"),
            edge.get("type") or edge.get("properties", {}).get("label", ""),
        )
        for edge in accumulated.get("edges", [])
    }
    
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
    
    return {"nodes": merged_nodes, "edges": merged_edges}
