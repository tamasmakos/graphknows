"""
Centrality Analysis Module.

Calculates various centrality measures for the graph, both globally and per-community.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_centrality_measures(
    graph: nx.Graph, 
    node_type_filter: Optional[str] = 'ENTITY_CONCEPT'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Degree, Betweenness, Closeness, and Eigenvector centrality.
    
    Args:
        graph: The NetworkX graph.
        node_type_filter: specific node type to consider (default: ENTITY_CONCEPT). 
                          If None, all nodes are considered.
                          
    Returns:
        Dict mapping centrality type to a dict of {node_id: score}.
    """
    # Create a subgraph view for the specific node type if requested
    # Note: Centrality often depends on the whole structure, but usually we only care about
    # how ENTITIES are central. If we filter the graph to ONLY entities, we miss the
    # paths through other node types (if any). 
    # However, in this ontology, ENTITY_CONCEPTs connect to ENTITY_CONCEPTs via relations.
    # Segments/Chunks/Docs are structural. We probably want the projection or subgraph of just entities.
    
    target_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == node_type_filter]
    
    if not target_nodes:
        logger.warning(f"No nodes found for type {node_type_filter}. Skipping centrality.")
        return {}

    subgraph = graph.subgraph(target_nodes)
    
    # Check if empty
    if subgraph.number_of_nodes() == 0:
        return {}

    logger.info(f"Calculating centrality for {subgraph.number_of_nodes()} nodes...")
    
    results = {}
    
    try:
        # 1. Degree Centrality
        results['degree'] = nx.degree_centrality(subgraph)
        
        # 2. Betweenness Centrality (can be slow, maybe limit k for approximation if large)
        # For now, we run full. If too slow, we can optimize.
        results['betweenness'] = nx.betweenness_centrality(subgraph, weight='weight')
        
        # 3. Closeness Centrality
        results['closeness'] = nx.closeness_centrality(subgraph)
        
        # 4. Eigenvector Centrality (max_iter increased for stability)
        try:
            results['eigenvector'] = nx.eigenvector_centrality(subgraph, max_iter=1000, weight='weight')
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge. Using PageRank instead.")
            results['eigenvector'] = nx.pagerank(subgraph, weight='weight')
            
        # 5. PageRank (Good for directed graphs)
        results['pagerank'] = nx.pagerank(subgraph, weight='weight')
        
    except Exception as e:
        logger.error(f"Error calculating centrality: {e}")
        
    return results

def get_top_entities_global(
    centrality_results: Dict[str, Dict[str, float]],
    graph: nx.Graph,
    top_k: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top K entities for each centrality measure globally.
    """
    stats = {}
    
    for measure, scores in centrality_results.items():
        # Sort by score desc
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        rankings = []
        for node_id, score in sorted_nodes:
            node_data = graph.nodes[node_id]
            rankings.append({
                "node_id": node_id,
                "name": node_data.get('name', node_id),
                "score": score,
                "description": node_data.get('description', '')[:50] + "..." if node_data.get('description') else ""
            })
        
        stats[measure] = rankings
        
    return stats

def get_top_entities_per_community(
    centrality_results: Dict[str, Dict[str, float]],
    communities: Dict[str, int],
    graph: nx.Graph,
    top_k: int = 5
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Get top K entities for each centrality measure, grouped by community.
    
    Returns:
        Dict[community_id, Dict[measure, List[NodeInfo]]]
    """
    # Invert community map: comm_id -> list of nodes
    comm_nodes = {}
    for node, comm_id in communities.items():
        if comm_id not in comm_nodes:
            comm_nodes[comm_id] = []
        comm_nodes[comm_id].append(node)
        
    results = {}
    
    for comm_id, nodes in comm_nodes.items():
        comm_stats = {}
        
        # Filter nodes to those in this community
        nodes_set = set(nodes)
        
        for measure, scores in centrality_results.items():
            # Filter scores for this community
            comm_scores = {n: s for n, s in scores.items() if n in nodes_set}
            
            if not comm_scores:
                continue
                
            # Sort
            sorted_nodes = sorted(comm_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            rankings = []
            for node_id, score in sorted_nodes:
                node_data = graph.nodes[node_id]
                rankings.append({
                    "node_id": node_id,
                    "name": node_data.get('name', node_id),
                    "score": score
                })
            
            comm_stats[measure] = rankings
            
        results[str(comm_id)] = comm_stats
        
    return results

def analyze_community_centrality_distribution(
    centrality_results: Dict[str, Dict[str, float]],
    communities: Dict[str, int]
) -> pd.DataFrame:
    """
    Create a DataFrame suitable for plotting distributions of centrality across communities.
    """
    rows = []
    
    for measure, scores in centrality_results.items():
        for node, score in scores.items():
            if node in communities:
                rows.append({
                    "node": node,
                    "community": communities[node],
                    "measure": measure,
                    "score": score
                })
                
    return pd.DataFrame(rows)
