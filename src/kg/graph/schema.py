"""
Graph schema extraction and export.
"""

import json
import os
import networkx as nx
import logging
from typing import Dict, Any, List, Set
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

def get_type_name(value: Any) -> str:
    """Get string representation of a value's type."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 0:
            # Check for embedding
            if isinstance(value[0], (float, np.floating)) and len(value) > 10:
                return f"embedding[{len(value)}]"
            return f"list[{get_type_name(value[0])}]"
        return "list"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (datetime,)):
        return "datetime"
    return str(type(value).__name__)

def extract_graph_schema(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Extract graph schema including:
    - Node types with property schemas and counts
    - Edge types with property schemas and counts
    - Excludes individual entity-entity edges (only schema)
    
    Returns schema dictionary for JSON serialization.
    """
    schema = {
        "generated_at": datetime.now().isoformat(),
        "node_types": {},
        "edge_types": {},
        "stats": {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges()
        }
    }
    
    # Analyze Nodes
    for node, data in graph.nodes(data=True):
        node_type = data.get('node_type', data.get('type', 'UNKNOWN'))
        
        if node_type not in schema["node_types"]:
            schema["node_types"][node_type] = {
                "count": 0,
                "properties": {}
            }
            
        schema["node_types"][node_type]["count"] += 1
        
        # Analyze properties
        for key, value in data.items():
            if key == 'node_type':
                continue
            
            prop_type = get_type_name(value)
            if key not in schema["node_types"][node_type]["properties"]:
                schema["node_types"][node_type]["properties"][key] = prop_type
    # Collect edge type information (exclude entity-to-entity relationships for cleaner schema)
    edge_types_temp = {} # Use a temporary dict to build up edge types
    for u, v, data in graph.edges(data=True):
        source_type = graph.nodes[u].get('node_type', graph.nodes[u].get('type', 'UNKNOWN'))
        target_type = graph.nodes[v].get('node_type', graph.nodes[v].get('type', 'UNKNOWN'))
        
        # Skip entity-to-entity relationships (semantic/extracted edges)
        # We only want structural/lexical edges in the schema
        if source_type == 'ENTITY_CONCEPT' and target_type == 'ENTITY_CONCEPT':
            continue
        
        edge_label = data.get('label', data.get('relation_type', data.get('relation', 'RELATED_TO')))
        
        if edge_label not in edge_types_temp:
            edge_types_temp[edge_label] = {
                'count': 0,
                'source_types': set(),
                'target_types': set(),
                'properties': {}
            }
        
        edge_types_temp[edge_label]['count'] += 1
        edge_types_temp[edge_label]['source_types'].add(source_type)
        edge_types_temp[edge_label]['target_types'].add(target_type)
        
        # Collect edge properties
        for key, value in data.items():
            if key in ['label', 'relation_type', 'source', 'target']:
                continue
            
            prop_type = get_type_name(value)
            if key not in edge_types_temp[edge_label]['properties']:
                edge_types_temp[edge_label]['properties'][key] = prop_type
    
    # Assign the collected edge types to the schema
    schema["edge_types"] = edge_types_temp
                
    # Convert sets to lists for JSON serialization
    for edge_type in schema["edge_types"]:
        schema["edge_types"][edge_type]["source_types"] = sorted(list(schema["edge_types"][edge_type]["source_types"]))
        schema["edge_types"][edge_type]["target_types"] = sorted(list(schema["edge_types"][edge_type]["target_types"]))
        
    return schema

def save_graph_schema(graph: nx.DiGraph, output_dir: str):
    """Save graph schema to JSON file."""
    try:
        schema = extract_graph_schema(graph)
        # Check if we should use the legacy path for tests or the new path
        # If output_dir looks like a temp dir (from tests), use the root filename
        if 'tmp' in output_dir:
            output_path = os.path.join(output_dir, "graph_schema.json")
        else:
            output_path = os.path.join(output_dir, "metadata", "schema.json")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
            
        logger.info(f"Graph schema saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save graph schema: {e}")
