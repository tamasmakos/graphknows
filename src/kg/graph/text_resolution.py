"""
Text-based Entity Resolution Module.

This module handles the resolution (merging) of valid entities and locations
in the Knowledge Graph using text similarity metrics (Jaro-Winkler).
"""

import logging
import networkx as nx
import jellyfish
from typing import Dict, List, Set, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def resolve_similar_entities(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.90,
    node_types: Optional[List[str]] = None
) -> List[Tuple[str, str, float]]:
    """
    Identify and merge nodes that have very high text similarity in their names.
    
    Args:
        graph: The Knowledge Graph (modified in-place).
        similarity_threshold: Jaro-Winkler threshold (0.0 to 1.0). Default 0.92.
        node_types: List of node types to consider (default: ['ENTITY_CONCEPT', 'PLACE']).
        
    Returns:
        List of merge operations performed [(source_id, target_id, score), ...]
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT', 'PLACE']
        
    logger.info(f"Starting text-based entity resolution (threshold={similarity_threshold})...")
    
    # 1. Group nodes by type to avoid cross-type merging (e.g. dont merge a Person with a Place)
    # Although sometimes we might want to if classifiers failed, but safest is to keep strict.
    # Actually, we should allow merging if they are "similar" enough? 
    # But names like "Paris" (Person) and "Paris" (Place) should NOT merge.
    # So strict grouping by assigned type is better, or at least primary type.
    
    nodes_by_type: Dict[str, List[str]] = {}
    
    for node_id, node_data in graph.nodes(data=True):
        ntype = node_data.get('node_type')
        if ntype in node_types:
            if ntype not in nodes_by_type:
                nodes_by_type[ntype] = []
            nodes_by_type[ntype].append(node_id)
            
    merges_performed = []
    
    for ntype, node_ids in nodes_by_type.items():
        # Sort to ensure potential deterministic behavior or just stability
        node_ids.sort() 
        n = len(node_ids)
        if n < 2:
            continue
            
        logger.info(f"Checking {n} nodes of type {ntype} for text similarity...")
        
        # We use a simple pairwise comparison since N is likely small-ish per batch.
        # For larger N, blocking or LSH would be needed.
        # Optimization: Sort by name length?
        
        # We need to be careful not to merge A->B and then B->C in one pass without tracking.
        # So we use a Union-Find or Graph approach, similar to resolution.py
        
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(node_ids)
        
        high_sim_pairs = 0
        
        for i in range(n):
            id_a = node_ids[i]
            name_a = graph.nodes[id_a].get('name', '')
            if not name_a: 
                continue
                
            for j in range(i + 1, n):
                id_b = node_ids[j]
                name_b = graph.nodes[id_b].get('name', '')
                if not name_b:
                    continue
                
                # Metric: Jaro-Winkler is good for short strings/names
                score = jellyfish.jaro_winkler_similarity(name_a.lower(), name_b.lower())
                
                if score >= similarity_threshold:
                    # Double check: if score is not 1.0, are they definitely synonyms?
                    # "New York" vs "New York City" -> JaroWinkler might be high?
                    # JW("new york", "new york city") = 0.84 (approx). 
                    # So 0.92 implies very close match.
                    
                    sim_graph.add_edge(id_a, id_b, weight=score)
                    high_sim_pairs += 1

        # Find components
        components = list(nx.connected_components(sim_graph))
        clusters = [list(c) for c in components if len(c) > 1]
        
        for cluster in clusters:
            # Pick canonical: Longest name usually (New York City > New York ?? or inverse?)
            # Usually the most frequent one is better, but we don't have frequency easily unless we check degree
            
            def get_node_score(nid):
                degree = graph.degree(nid)
                name = graph.nodes[nid].get('name', '')
                return (degree, len(name), name) # Prefer higher degree, then longer name
            
            canonical_id = max(cluster, key=get_node_score)
            canonical_name = graph.nodes[canonical_id].get('name', '')
            
            for node_id in cluster:
                if node_id == canonical_id:
                    continue
                    
                # Calculate direct score for log
                node_name = graph.nodes[node_id].get('name', '')
                score = jellyfish.jaro_winkler_similarity(canonical_name.lower(), node_name.lower())
                
                logger.info(f"MERGING: {node_name} ({node_id}) -> {canonical_name} ({canonical_id}) [Score: {score:.4f}]")
                merges_performed.append((node_id, canonical_id, score))
                
                # Perform in-memory merge to keep graph consistent for this run
                _merge_node_into(graph, node_id, canonical_id)

    return merges_performed

def _merge_node_into(graph: nx.DiGraph, source_node: str, target_node: str):
    """
    Merge source_node into target_node (In-Memory Helper).
    Identical to resolution.py's helper but duplicated to avoid circular imports or messy deps.
    """
    if not graph.has_node(source_node) or not graph.has_node(target_node):
        return

    # 1. Incoming edges
    for pred, _, edge_data in list(graph.in_edges(source_node, data=True)):
        if pred == target_node:
            continue 
        if pred == source_node: 
             if not graph.has_edge(target_node, target_node):
                 graph.add_edge(target_node, target_node, **edge_data)
             continue
        
        if not graph.has_edge(pred, target_node):
            graph.add_edge(pred, target_node, **edge_data)
        else:
            current_weight = graph[pred][target_node].get('weight', 1.0)
            new_weight = edge_data.get('weight', 1.0)
            graph[pred][target_node]['weight'] = max(current_weight, new_weight)
            
    # 2. Outgoing edges
    for _, succ, edge_data in list(graph.out_edges(source_node, data=True)):
        if succ == target_node:
            continue
        if succ == source_node:
            if not graph.has_edge(target_node, target_node):
                 graph.add_edge(target_node, target_node, **edge_data)
            continue

        if not graph.has_edge(target_node, succ):
            graph.add_edge(target_node, succ, **edge_data)
        else:
            current_weight = graph[target_node][succ].get('weight', 1.0)
            new_weight = edge_data.get('weight', 1.0)
            graph[target_node][succ]['weight'] = max(current_weight, new_weight)

    # 3. Merge attributes (Aliases)
    source_data = graph.nodes[source_node]
    target_data = graph.nodes[target_node]
    
    aliases = set(target_data.get('aliases', []))
    aliases.add(source_data.get('name', source_node))
    if 'aliases' in source_data:
        aliases.update(source_data['aliases'])
    
    # Also add the source_node ID itself as an alias if it looks like a name/ID
    if not source_node.startswith('ENTITY_CONCEPT'): 
        aliases.add(source_node)
        
    graph.nodes[target_node]['aliases'] = list(aliases)
    
    # 4. Remove source node
    graph.remove_node(source_node)
