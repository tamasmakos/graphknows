"""
Embedding Similarity Edge Computation for Knowledge Graph.

Computes cosine similarity between node embeddings and adds similarity edges
between nodes that exceed a threshold. This enriches the graph structure
for better community detection.

Key features:
- Computes pairwise cosine similarity between entity embeddings
- Adds new SIMILAR_TO edges between highly similar nodes
- Sets edge weight to similarity score for weighted community detection
- Can optionally update existing edge weights
"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)





def compute_embedding_similarity_matrix(
    embeddings: Dict[str, np.ndarray]
) -> Tuple[List[str], np.ndarray]:
    """
    Compute pairwise cosine similarity matrix for all embeddings.
    
    Args:
        embeddings: Dictionary mapping node_id to embedding vector
        
    Returns:
        Tuple of (node_ids list, similarity matrix)
    """
    node_ids = list(embeddings.keys())
    n = len(node_ids)
    
    if n == 0:
        return [], np.array([])
    
    # Stack embeddings into matrix
    embedding_matrix = np.array([embeddings[nid] for nid in node_ids])
    
    # Normalize embeddings
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embedding_matrix / norms
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return node_ids, similarity_matrix


def compute_embedding_similarity_edges(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.7,
    node_types: Optional[List[str]] = None,
    add_new_edges: bool = True,
    update_existing_weights: bool = True,
    edge_label: str = "SIMILAR_TO"
) -> Dict[str, int]:
    """
    Compute embedding similarity and add/update edges in the graph.
    
    Finds pairs of nodes with high embedding similarity and:
    1. Adds new SIMILAR_TO edges between unconnected similar nodes
    2. Optionally updates existing edge weights with similarity scores
    
    This is designed to be called AFTER embedding generation and BEFORE
    community detection, so that Leiden algorithm can use similarity-weighted edges.
    
    Args:
        graph: NetworkX DiGraph with embeddings stored on nodes
        similarity_threshold: Minimum similarity to create/update edge (default: 0.7)
        node_types: Node types to consider (default: ['ENTITY_CONCEPT'])
        add_new_edges: Whether to add new edges for similar unconnected nodes
        update_existing_weights: Whether to update weights on existing edges
        edge_label: Label for new similarity edges
        
    Returns:
        Statistics dictionary with counts of operations
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT']
    
    logger.info(f"Computing embedding similarity edges for node types: {node_types}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    
    # Collect nodes with embeddings
    embeddings: Dict[str, np.ndarray] = {}
    
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type', '')
        if node_type not in node_types:
            continue
        
        embedding = node_data.get('embedding')
        if embedding is not None:
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            embeddings[node_id] = embedding
    
    if len(embeddings) < 2:
        logger.warning(f"Not enough nodes with embeddings for similarity computation (found {len(embeddings)})")
        return {'nodes_with_embeddings': len(embeddings), 'edges_added': 0, 'weights_updated': 0}
    
    logger.info(f"Found {len(embeddings)} nodes with embeddings")
    
    # Compute similarity matrix
    node_ids, similarity_matrix = compute_embedding_similarity_matrix(embeddings)
    n = len(node_ids)
    
    # Track existing edges between these nodes
    existing_edges: Set[Tuple[str, str]] = set()
    for i, node1 in enumerate(node_ids):
        for j, node2 in enumerate(node_ids):
            if i != j and graph.has_edge(node1, node2):
                existing_edges.add((node1, node2))
    
    edges_added = 0
    weights_updated = 0
    high_similarity_pairs = 0
    
    # Process similarity matrix
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only (avoid duplicates)
            similarity = similarity_matrix[i, j]
            
            if similarity >= similarity_threshold:
                high_similarity_pairs += 1
                node1, node2 = node_ids[i], node_ids[j]
                
                # Check if edge exists in either direction
                has_edge_1_2 = graph.has_edge(node1, node2)
                has_edge_2_1 = graph.has_edge(node2, node1)
                
                if not has_edge_1_2 and not has_edge_2_1:
                    # Add new similarity edge
                    if add_new_edges:
                        graph.add_edge(
                            node1, node2,
                            label=edge_label,
                            relation_type=edge_label,
                            graph_type="similarity",
                            weight=float(similarity),
                            similarity_score=float(similarity)
                        )
                        edges_added += 1
                else:
                    # Update existing edge weight
                    if update_existing_weights:
                        if has_edge_1_2:
                            # Combine with existing weight if present
                            existing_weight = graph[node1][node2].get('weight', 1.0)
                            # Use max of existing weight and similarity
                            new_weight = max(existing_weight, float(similarity))
                            graph[node1][node2]['weight'] = new_weight
                            graph[node1][node2]['similarity_score'] = float(similarity)
                            weights_updated += 1
                        if has_edge_2_1:
                            existing_weight = graph[node2][node1].get('weight', 1.0)
                            new_weight = max(existing_weight, float(similarity))
                            graph[node2][node1]['weight'] = new_weight
                            graph[node2][node1]['similarity_score'] = float(similarity)
                            weights_updated += 1
    
    stats = {
        'nodes_with_embeddings': len(embeddings),
        'high_similarity_pairs': high_similarity_pairs,
        'edges_added': edges_added,
        'weights_updated': weights_updated,
        'similarity_threshold': similarity_threshold,
    }
    
    logger.info(f"Similarity computation complete:")
    logger.info(f"  - High similarity pairs (>= {similarity_threshold}): {high_similarity_pairs}")
    logger.info(f"  - New edges added: {edges_added}")
    logger.info(f"  - Edge weights updated: {weights_updated}")
    
    return stats



