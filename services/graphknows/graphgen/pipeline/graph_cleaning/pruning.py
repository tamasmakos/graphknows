import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)

def prune_graph(graph: nx.DiGraph, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prune the graph based on degree or other metrics.
    Simple implementation for now.
    """
    threshold = config.get('pruning_threshold', 0.0)
    initial_nodes = graph.number_of_nodes()
    initial_edges = graph.number_of_edges()
    
    # 1. Prune Low-Confidence Edges
    edges_removed = 0
    min_confidence = config.get('min_edge_confidence', 0.0)
    
    if min_confidence > 0:
        edges_to_remove = []
        for u, v, data in graph.edges(data=True):
            # Only prune entity_relation edges, not others (like structural ones)
            if data.get('graph_type') == 'entity_relation':
                conf = data.get('confidence', 1.0)
                if conf < min_confidence:
                    edges_to_remove.append((u, v))
        
        graph.remove_edges_from(edges_to_remove)
        edges_removed = len(edges_to_remove)
        logger.info(f"Pruned {edges_removed} edges with confidence < {min_confidence}")

    # 2. Prune isolated nodes if configured
    nodes_to_remove = []
    if config.get('prune_isolated_nodes', True):
        nodes_to_remove = [n for n in graph.nodes() if graph.degree(n) == 0]
        
    graph.remove_nodes_from(nodes_to_remove)
    
    return {
        "nodes_removed": len(nodes_to_remove),
        "edges_removed": edges_removed,
        "final_nodes": graph.number_of_nodes(),
        "final_edges": graph.number_of_edges()
    }
