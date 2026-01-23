import networkx as nx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_output_directory(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

def merge_node_into(graph: nx.DiGraph, source_node: str, target_node: str):
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
    if not source_node.startswith('ENTITY_') and not source_node.startswith('PLACE_'): 
        aliases.add(source_node)
        
    graph.nodes[target_node]['aliases'] = list(aliases)
    
    # 4. Remove source node
    graph.remove_node(source_node)

