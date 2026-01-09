"""
Semantic Entity Resolution Module.

This module handles the resolution (merging) of semantically similar entities
in the Knowledge Graph using their vector embeddings.
"""

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional

from src.kg.graph.similarity import compute_embedding_similarity_matrix

logger = logging.getLogger(__name__)

def merge_similar_nodes(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.95,
    node_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Identify and merge nodes that have very high embedding similarity.
    
    Algorithm:
    1. Identify target nodes (ENTITY_CONCEPT with embeddings).
    2. Compute pairwise similarity matrix.
    3. Build a temporary similarity graph where edges exist if sim > threshold.
    4. Find connected components (clusters of identical entities).
    5. For each cluster, merge nodes into a canonical node.
    
    Args:
        graph: The Knowledge Graph (modified in-place).
        similarity_threshold: Threshold above which nodes are considered identical.
        node_types: List of node types to consider (default: ['ENTITY_CONCEPT']).
        
    Returns:
        Statistics about the operation.
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT']
        
    logger.info(f"Starting semantic entity resolution (threshold={similarity_threshold})...")
    
    # 1. Collect nodes with embeddings
    embeddings: Dict[str, np.ndarray] = {}
    node_data_map: Dict[str, Dict] = {}
    
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') in node_types:
            embedding = node_data.get('embedding')
            if embedding is not None:
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                embeddings[node_id] = embedding
                node_data_map[node_id] = node_data

    if len(embeddings) < 2:
        logger.info("Not enough nodes with embeddings for resolution.")
        return {'merged_nodes': 0, 'clusters_found': 0}

    # 2. Compute similarity matrix
    node_ids, similarity_matrix = compute_embedding_similarity_matrix(embeddings)
    n = len(node_ids)
    
    # 3. Build similarity graph (undirected)
    sim_graph = nx.Graph()
    sim_graph.add_nodes_from(node_ids)
    
    high_sim_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                sim_graph.add_edge(node_ids[i], node_ids[j], weight=similarity_matrix[i, j])
                high_sim_pairs += 1
                
    # 4. Find connected components (clusters)
    components = list(nx.connected_components(sim_graph))
    clusters = [list(c) for c in components if len(c) > 1]
    
    logger.info(f"Found {len(clusters)} clusters of similar entities (from {high_sim_pairs} pairs).")
    
    # 5. Merge nodes
    nodes_merged = 0
    
    for cluster in clusters:
        # Select canonical node
        # Heuristic: Prefer node with highest degree, then longest name (usually more descriptive)
        # We can also check if one is already a "canonical" form from previous steps, but we don't have that info easily.
        
        def get_node_score(nid):
            degree = graph.degree(nid)
            name_len = len(graph.nodes[nid].get('name', ''))
            return (degree, name_len)
            
        canonical_id = max(cluster, key=get_node_score)
        
        # Log resolution details
        cluster_names = [graph.nodes[n].get('name', n) for n in cluster]
        logger.info(f"Resolving cluster to '{graph.nodes[canonical_id].get('name', canonical_id)}': {cluster_names}")

        # Merge others into canonical
        for node_id in cluster:
            if node_id == canonical_id:
                continue
                
            _merge_node_into(graph, source_node=node_id, target_node=canonical_id)
            nodes_merged += 1
            
    logger.info(f"Semantic resolution complete. Merged {nodes_merged} nodes into {len(clusters)} canonical entities.")
    
    return {
        'merged_nodes': nodes_merged,
        'clusters_found': len(clusters),
        'high_similarity_pairs': high_sim_pairs
    }

def _merge_node_into(graph: nx.DiGraph, source_node: str, target_node: str):
    """
    Merge source_node into target_node.
    
    1. Move all incoming edges from source to target.
    2. Move all outgoing edges from source to target.
    3. Copy relevant attributes/aliases.
    4. Remove source_node.
    """
    if not graph.has_node(source_node) or not graph.has_node(target_node):
        return

    # 1. Incoming edges
    for pred, _, edge_data in list(graph.in_edges(source_node, data=True)):
        if pred == target_node:
            continue # Don't create self-loops if they were connected
            
        if pred == source_node: # Handle self-loops on source
             if not graph.has_edge(target_node, target_node):
                 graph.add_edge(target_node, target_node, **edge_data)
             continue
        
        # If edge already exists, we might want to merge weights, but for now just ensure connectivity
        if not graph.has_edge(pred, target_node):
            graph.add_edge(pred, target_node, **edge_data)
        else:
            # Update weight if applicable
            current_weight = graph[pred][target_node].get('weight', 1.0)
            new_weight = edge_data.get('weight', 1.0)
            graph[pred][target_node]['weight'] = max(current_weight, new_weight)
            
    # 2. Outgoing edges
    for _, succ, edge_data in list(graph.out_edges(source_node, data=True)):
        if succ == target_node:
            continue
            
        if succ == source_node: # Handle self-loops on source
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
    
    # Handle aliases
    aliases = set(target_data.get('aliases', []))
    aliases.add(source_data.get('name', source_node))
    if 'aliases' in source_data:
        aliases.update(source_data['aliases'])
    
    # Also add the source_node ID itself as an alias if it looks like a name
    if not source_node.startswith('ENTITY_'): # If IDs are names
        aliases.add(source_node)
        
    graph.nodes[target_node]['aliases'] = list(aliases)
    
    # 4. Remove source node
    graph.remove_node(source_node)
