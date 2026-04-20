"""
Graph Pruning Module.

Handles removing low-quality or irrelevant nodes and edges from the graph.
"""

import logging
import networkx as nx
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def prune_graph(graph: nx.DiGraph, config: Dict[str, Any]) -> Dict[str, int]:
    """
    Prune edges and nodes from the graph based on configuration.
    
    Operations:
    1. Remove edges with weight below threshold (if weights exist).
    2. Remove isolated nodes (optional).
    
    Args:
        graph: NetworkX graph to prune (modified in-place)
        config: Configuration dictionary (usually from 'incremental' section)
        
    Returns:
        Statistics about pruned items
    """
    stats = {
        'edges_pruned': 0,
        'nodes_pruned': 0
    }
    
    if not config.get('enable_pruning', True):
        return stats
        
    threshold = config.get('pruning_threshold', 0.01)
    prune_isolated = config.get('prune_isolated_nodes', True)
    
    logger.info(f"Taking a laser scan... Pruning graph (threshold={threshold}, prune_isolated={prune_isolated})")
    
    # 1. Prune Edges by Weight
    edges_to_remove = []
    dropped_logs = []
    
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight')
        if weight is not None:
             try:
                 w_val = float(weight)
                 if w_val < threshold:
                     edges_to_remove.append((u, v))
                     if len(dropped_logs) < 10: # Sample logs
                         dropped_logs.append(f"{u} -> {v} (weight={w_val:.4f})")
             except (ValueError, TypeError):
                 pass
                 
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)
        stats['edges_pruned'] = len(edges_to_remove)
        logger.info(f"✂️  Pruned {len(edges_to_remove)} edges with weight < {threshold}")
        if dropped_logs:
            logger.debug("Sample pruned edges:\n - " + "\n - ".join(dropped_logs))
    
    # 2. Prune Isolated Nodes
    vital_types = {'DAY', 'SEGMENT', 'EPISODE', 'TOPIC', 'SUBTOPIC'} # Vital types to preserve
    
    if prune_isolated:
        nodes_to_remove = []
        for node in list(graph.nodes()):
            # Check degree (in + out)
            if graph.degree(node) == 0:
                # Protect vital types
                if graph.nodes[node].get('node_type') in vital_types:
                    continue
                    
                nodes_to_remove.append(node)
        
        if nodes_to_remove:
            graph.remove_nodes_from(nodes_to_remove)
            stats['nodes_pruned'] = len(nodes_to_remove)
            logger.info(f"✂️  Pruned {len(nodes_to_remove)} isolated nodes")
            if len(nodes_to_remove) > 0:
                 logger.debug(f"Sample pruned nodes: {nodes_to_remove[:5]}")
    # 3. Prune Disconnected Components (Irrelevant Subgraphs)
    # Remove small components that are not connected to the main backbone (Day/Segment nodes)
    
    # Use weakly connected components for directed graph
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
        
    min_component_size = config.get('min_component_size', 3)
    # Vital types that anchor a component
    vital_types = {'DAY', 'SEGMENT', 'EPISODE', 'TOPIC', 'SUBTOPIC'} 
    
    components_to_remove = []
    
    for comp in components:
        # If component is very small
        if len(comp) < min_component_size:
            # Check if it contains any vital node
            has_vital = False
            for node in comp:
                ntype = graph.nodes[node].get('node_type')
                if ntype in vital_types:
                    has_vital = True
                    break
            
            if not has_vital:
                components_to_remove.append(comp)
                
    if components_to_remove:
        nodes_removed = 0
        for comp in components_to_remove:
            graph.remove_nodes_from(comp)
            nodes_removed += len(comp)
            
        stats['nodes_pruned'] += nodes_removed
        logger.info(f"✂️  Pruned {len(components_to_remove)} small disconnected components (total {nodes_removed} nodes)")

    return stats
