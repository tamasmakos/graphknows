from dataclasses import dataclass
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class CommunityQualityMetrics:
    """Metrics for evaluating community quality and balance"""
    modularity: float
    size_variance: float
    avg_size: float
    min_size: int
    max_size: int
    size_ratio: float  # max_size / min_size
    text_length_variance: float
    optimal_for_summarization: bool

def calculate_simple_modularity(graph: nx.DiGraph, communities: Dict[str, int]) -> float:
    """Calculate modularity for the entity subgraph using NetworkX's implementation.

    This replaces the prior ratio-of-internal-edges approximation with a proper modularity score.
    """
    try:
        # Extract entity subgraph and convert to undirected for modularity
        entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT']
        if not entity_nodes:
            return 0.0
        undirected_graph = graph.subgraph(entity_nodes).to_undirected()
        if undirected_graph.number_of_edges() == 0:
            return 0.0

        # Build community sets aligned to nodes actually present in the subgraph
        community_sets: List[Set[str]] = []
        for comm_id in set(communities.values()):
            nodes_in_comm = {n for n, cid in communities.items() if cid == comm_id and undirected_graph.has_node(n)}
            if nodes_in_comm:
                community_sets.append(nodes_in_comm)
        if not community_sets:
            return 0.0

        return float(nx.algorithms.community.modularity(undirected_graph, community_sets))
    except Exception as e:
        logger.warning(f"Could not calculate modularity: {e}")
        return 0.0

def evaluate_community_quality(graph: nx.DiGraph, communities: Dict[str, int]) -> CommunityQualityMetrics:
    """Evaluate community quality for summarization purposes"""
    
    def calculate_text_length_for_community(graph: nx.DiGraph, entity_ids: List[str]) -> int:
        """Calculate total text length for entities in a community"""
        total_chars = 0
        
        for entity_id in entity_ids:
            # Find chunks connected to this entity
            for predecessor in graph.predecessors(entity_id):
                edge_data = graph.get_edge_data(predecessor, entity_id)
                if (edge_data and 
                    edge_data.get('label') == 'HAS_ENTITY' and
                    graph.nodes[predecessor].get('node_type') == 'CHUNK'):
                    
                    # Get text from sentences attribute
                    chunk_data = graph.nodes[predecessor]
                    if 'sentences' in chunk_data:
                        sentences = chunk_data['sentences']
                        if isinstance(sentences, list):
                            chunk_text = ' '.join(sentences)
                        else:
                            chunk_text = str(sentences)
                        total_chars += len(chunk_text)
        
        return total_chars
    
    # Group entities by community
    from collections import defaultdict
    community_sizes = defaultdict(int)
    community_text_lengths = defaultdict(int)
    
    for entity_id, comm_id in communities.items():
        community_sizes[comm_id] += 1
        # Calculate text length for this entity's chunks
        text_length = calculate_text_length_for_community(graph, [entity_id])
        community_text_lengths[comm_id] += text_length
    
    sizes = list(community_sizes.values())
    text_lengths = list(community_text_lengths.values())
    
    if not sizes:
        return CommunityQualityMetrics(0, 0, 0, 0, 0, 1, 0, False)
    
    # Calculate basic metrics
    avg_size = np.mean(sizes)
    size_variance = np.var(sizes) if len(sizes) > 1 else 0
    min_size = min(sizes)
    max_size = max(sizes)
    size_ratio = max_size / min_size if min_size > 0 else float('inf')
    
    text_length_variance = np.var(text_lengths) if len(text_lengths) > 1 else 0
    
    # Check if communities are optimal for summarization
    # Ideal: 10K-100K characters per community, not too imbalanced
    optimal_communities = sum(1 for length in text_lengths 
                            if 10000 <= length <= 100000)
    total_communities = len(text_lengths)
    optimal_ratio = optimal_communities / total_communities if total_communities > 0 else 0
    
    optimal_for_summarization = (
        optimal_ratio >= 0.7 and  # At least 70% in optimal range
        size_ratio <= 10 and     # Not too imbalanced
        max(text_lengths) <= 200000  # No extremely large communities
    )
    
    # Calculate modularity (simplified approximation)
    modularity = calculate_simple_modularity(graph, communities)
    
    return CommunityQualityMetrics(
        modularity=modularity,
        size_variance=size_variance,
        avg_size=avg_size,
        min_size=min_size,
        max_size=max_size,
        size_ratio=size_ratio,
        text_length_variance=text_length_variance,
        optimal_for_summarization=optimal_for_summarization
    )
