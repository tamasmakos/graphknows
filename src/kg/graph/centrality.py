import networkx as nx
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def generate_centrality_description(
    measure_name: str,
    entity_name: str,
    z_score: float,
    score: float,
    mean: float
) -> str:
    """
    Generate a human-readable description of an entity's centrality position.
    
    Args:
        measure_name: Name of the centrality measure (e.g., 'degree', 'betweenness')
        entity_name: Name of the entity
        z_score: Standardized score (z-score) for the measure
        score: Raw centrality score
        mean: Mean centrality score for all entities
        
    Returns:
        Human-readable description string
    """
    # Determine significance level based on z-score
    if abs(z_score) < 0.5:
        relative_term = "relatively average"
    elif abs(z_score) < 1.0:
        relative_term = "somewhat" if z_score > 0 else "somewhat less"
    elif abs(z_score) < 2.0:
        relative_term = "highly" if z_score > 0 else "relatively unimportant"
    else:
        relative_term = "extremely" if z_score > 0 else "very low"
    
    # Measure-specific descriptions
    descriptions = {
        'degree': {
            'high': f"{entity_name} is {relative_term} important in the network because it has many direct connections to other entities.",
            'low': f"{entity_name} has relatively few direct connections, making it less central in the network structure.",
            'average': f"{entity_name} has an average number of connections, representing typical network connectivity."
        },
        'betweenness': {
            'high': f"{entity_name} is {relative_term} important in the network because it acts as a bridge between many other entities, facilitating information flow.",
            'low': f"{entity_name} does not serve as a bridge between other entities, indicating it is not on critical paths in the network.",
            'average': f"{entity_name} occasionally acts as a bridge between entities, with typical betweenness centrality."
        },
        'closeness': {
            'high': f"{entity_name} is {relative_term} important in the network because it can reach all other entities through relatively short paths, making it well-positioned for information dissemination.",
            'low': f"{entity_name} is relatively distant from other entities in the network, requiring more steps to reach others.",
            'average': f"{entity_name} has average proximity to other entities in the network."
        },
        'eigenvector': {
            'high': f"{entity_name} is {relative_term} important in the network because it is connected to other highly connected entities, indicating it is part of a central cluster.",
            'low': f"{entity_name} is connected to entities that are themselves less central, placing it in a peripheral position.",
            'average': f"{entity_name} is connected to entities with average centrality, reflecting typical network positioning."
        },
        'pagerank': {
            'high': f"{entity_name} is {relative_term} important in the network because it receives connections from other important entities, indicating high influence and prominence.",
            'low': f"{entity_name} has low influence in the network, receiving few connections from important entities.",
            'average': f"{entity_name} has average influence, receiving typical connections from entities of average importance."
        },
        'harmonic': {
            'high': f"{entity_name} is {relative_term} important in the network because it has short average distances to all other entities, enabling efficient communication.",
            'low': f"{entity_name} has longer average distances to other entities, making it less accessible in the network.",
            'average': f"{entity_name} has average harmonic centrality, indicating typical accessibility to other entities."
        },
        'load': {
            'high': f"{entity_name} is {relative_term} important in the network because it carries a significant load of shortest paths between other entities.",
            'low': f"{entity_name} carries minimal load in the network's shortest paths, indicating peripheral positioning.",
            'average': f"{entity_name} carries an average load of shortest paths in the network."
        },
        'current_flow_betweenness': {
            'high': f"{entity_name} is {relative_term} important in the network because it acts as a critical bridge in multiple communication pathways between entities.",
            'low': f"{entity_name} is not a critical bridge in the network's communication pathways.",
            'average': f"{entity_name} has average importance as a bridge in network communication pathways."
        },
        'current_flow_closeness': {
            'high': f"{entity_name} is {relative_term} important in the network because it has efficient access to all other entities through multiple pathways.",
            'low': f"{entity_name} has limited efficient access to other entities through network pathways.",
            'average': f"{entity_name} has average efficient access to other entities in the network."
        },
        'katz': {
            'high': f"{entity_name} is {relative_term} important in the network because it is connected to influential entities, amplifying its own influence.",
            'low': f"{entity_name} has limited influence due to connections to less influential entities.",
            'average': f"{entity_name} has average influence through connections to entities of typical importance."
        }
    }
    
    # Select description based on z-score
    if z_score > 1.0:
        level = 'high'
    elif z_score < -1.0:
        level = 'low'
    else:
        level = 'average'
    
    # Get measure-specific description
    measure_descriptions = descriptions.get(measure_name, {
        'high': f"{entity_name} is {relative_term} important in the network based on {measure_name} centrality.",
        'low': f"{entity_name} has relatively low importance in the network based on {measure_name} centrality.",
        'average': f"{entity_name} has average importance in the network based on {measure_name} centrality."
    })
    
    return measure_descriptions.get(level, measure_descriptions['average'])

def calculate_entity_relation_centrality_measures(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate comprehensive centrality measures for all entity-relation nodes.
    
    This function calculates multiple centrality measures for nodes with graph_type = "entity_relation",
    excluding community, lexical graph, speaker, and speech nodes.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        
    Returns:
        Dictionary containing centrality statistics and results
    """
    logger.info("=== CALCULATING CENTRALITY MEASURES FOR ENTITY-RELATION NODES ===")
    
    # Get all entity-relation nodes (entities/concepts)
    entity_relation_nodes = [
        node_id for node_id, node_data in graph.nodes(data=True)
        if node_data.get('graph_type') == 'entity_relation'
    ]
    
    logger.info(f"Found {len(entity_relation_nodes)} entity-relation nodes for centrality calculation")
    
    if len(entity_relation_nodes) < 2:
        logger.warning("Not enough entity-relation nodes for centrality calculation (need at least 2)")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': 'Insufficient nodes for centrality calculation'
        }
    
    # Create subgraph with only entity-relation nodes and their connections
    entity_subgraph = graph.subgraph(entity_relation_nodes).copy()
    
    # Convert to undirected for centrality calculations (most measures work on undirected graphs)
    undirected_graph = entity_subgraph.to_undirected()
    
    if undirected_graph.number_of_edges() == 0:
        logger.warning("No edges found in entity-relation subgraph")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': 'No edges in entity-relation subgraph'
        }
    
    logger.info(f"Entity-relation subgraph: {undirected_graph.number_of_nodes()} nodes, {undirected_graph.number_of_edges()} edges")
    
    centrality_results = {}
    statistics = {}
    
    try:
        # 1. Degree Centrality
        logger.info("Calculating degree centrality...")
        degree_centrality = nx.degree_centrality(undirected_graph)
        centrality_results['degree'] = degree_centrality
        
        # 2. Betweenness Centrality
        logger.info("Calculating betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(undirected_graph, normalized=True)
        centrality_results['betweenness'] = betweenness_centrality
        
        # 3. Closeness Centrality
        logger.info("Calculating closeness centrality...")
        closeness_centrality = nx.closeness_centrality(undirected_graph)
        centrality_results['closeness'] = closeness_centrality
        
        # 4. Eigenvector Centrality
        logger.info("Calculating eigenvector centrality...")
        try:
            eigenvector_centrality = nx.eigenvector_centrality(undirected_graph, max_iter=1000, tol=1e-06)
            centrality_results['eigenvector'] = eigenvector_centrality
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, using Katz centrality as fallback")
            try:
                katz_centrality = nx.katz_centrality(undirected_graph, max_iter=1000, tol=1e-06)
                centrality_results['katz'] = katz_centrality
            except nx.PowerIterationFailedConvergence:
                logger.warning("Katz centrality also failed to converge, skipping eigenvector-like measures")
        
        # 5. PageRank Centrality
        logger.info("Calculating PageRank centrality...")
        pagerank_centrality = nx.pagerank(undirected_graph, alpha=0.85, max_iter=1000, tol=1e-06)
        centrality_results['pagerank'] = pagerank_centrality
        
        # 6. Harmonic Centrality
        logger.info("Calculating harmonic centrality...")
        harmonic_centrality = nx.harmonic_centrality(undirected_graph)
        centrality_results['harmonic'] = harmonic_centrality
        
        # 7. Load Centrality (alternative to betweenness)
        logger.info("Calculating load centrality...")
        try:
            load_centrality = nx.load_centrality(undirected_graph)
            centrality_results['load'] = load_centrality
        except Exception as e:
            logger.warning(f"Load centrality calculation failed: {e}")
        
        # 8. Current Flow Betweenness (for weighted graphs)
        logger.info("Calculating current flow betweenness centrality...")
        try:
            current_flow_betweenness = nx.current_flow_betweenness_centrality(undirected_graph)
            centrality_results['current_flow_betweenness'] = current_flow_betweenness
        except Exception as e:
            logger.warning(f"Current flow betweenness calculation failed: {e}")
        
        # 9. Current Flow Closeness
        logger.info("Calculating current flow closeness centrality...")
        try:
            current_flow_closeness = nx.current_flow_closeness_centrality(undirected_graph)
            centrality_results['current_flow_closeness'] = current_flow_closeness
        except Exception as e:
            logger.warning(f"Current flow closeness calculation failed: {e}")
        
        # Calculate statistics for each centrality measure first (needed for distance calculations)
        logger.info("Calculating centrality statistics...")
        for measure_name, measure_scores in centrality_results.items():
            if measure_scores:
                scores = list(measure_scores.values())
                statistics[measure_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75))
                }
        
        # Add centrality measures to graph nodes along with distance from average and descriptions
        logger.info("Adding centrality measures, distance from average, and descriptions to graph nodes...")
        for node_id in entity_relation_nodes:
            node_data = graph.nodes[node_id]
            
            # Get entity name for descriptions (fallback to node_id if name not available)
            entity_name = node_data.get('name', node_data.get('entity_name', str(node_id)))
            
            # Collect descriptions for summary
            centrality_descriptions = []
            z_scores_by_measure = {}
            
            # Add each centrality measure to the node along with distance from mean
            for measure_name, measure_scores in centrality_results.items():
                if node_id in measure_scores:
                    score = float(measure_scores[node_id])
                    node_data[f'{measure_name}_centrality'] = score
                    
                    # Calculate distance from average if statistics are available
                    if measure_name in statistics:
                        mean = statistics[measure_name]['mean']
                        std = statistics[measure_name]['std']
                        
                        # Absolute distance from mean
                        distance_from_mean = abs(score - mean)
                        node_data[f'{measure_name}_centrality_distance_from_mean'] = float(distance_from_mean)
                        
                        # Signed deviation from mean (positive = above average, negative = below average)
                        deviation_from_mean = score - mean
                        node_data[f'{measure_name}_centrality_deviation_from_mean'] = float(deviation_from_mean)
                        
                        # Standardized score (z-score) if std > 0
                        if std > 0:
                            z_score = (score - mean) / std
                            node_data[f'{measure_name}_centrality_z_score'] = float(z_score)
                        else:
                            z_score = 0.0
                            node_data[f'{measure_name}_centrality_z_score'] = 0.0
                        
                        z_scores_by_measure[measure_name] = z_score
                        
                        # Generate rule-based description
                        description = generate_centrality_description(
                            measure_name=measure_name,
                            entity_name=entity_name,
                            z_score=z_score,
                            score=score,
                            mean=mean
                        )
                        node_data[f'{measure_name}_centrality_description'] = description
                        centrality_descriptions.append(description)
            
            # Generate combined centrality summary
            if centrality_descriptions:
                # Count high/low/average measures
                high_count = sum(1 for z in z_scores_by_measure.values() if z > 1.0)
                low_count = sum(1 for z in z_scores_by_measure.values() if z < -1.0)
                avg_count = len(z_scores_by_measure) - high_count - low_count
                
                # Create summary based on overall pattern
                if high_count > low_count and high_count >= 2:
                    summary = f"{entity_name} is highly important in the network, showing strong centrality across multiple measures."
                elif low_count > high_count and low_count >= 2:
                    summary = f"{entity_name} has relatively low importance in the network, with below-average centrality across multiple measures."
                elif high_count == 1 and low_count == 0:
                    # Find which measure is high
                    high_measure = next((m for m, z in z_scores_by_measure.items() if z > 1.0), None)
                    if high_measure:
                        measure_label = high_measure.replace('_', ' ').title()
                        summary = f"{entity_name} shows particular strength in {measure_label} centrality, indicating specialized network importance."
                    else:
                        summary = f"{entity_name} has average to moderate importance in the network."
                else:
                    summary = f"{entity_name} has mixed centrality characteristics, with some measures above average and others below."
                
                node_data['centrality_summary'] = summary
                node_data['centrality_high_measures'] = high_count
                node_data['centrality_low_measures'] = low_count
                node_data['centrality_average_measures'] = avg_count
        
        # Find top nodes for each centrality measure
        top_nodes = {}
        for measure_name, measure_scores in centrality_results.items():
            if measure_scores:
                sorted_nodes = sorted(measure_scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes[measure_name] = sorted_nodes[:10]  # Top 10 nodes
        
        logger.info("✅ Centrality measures calculated successfully")
        logger.info(f"Processed {len(entity_relation_nodes)} entity-relation nodes")
        logger.info(f"Calculated {len(centrality_results)} centrality measures")
        
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': centrality_results,
            'statistics': statistics,
            'top_nodes': top_nodes,
            'graph_info': {
                'nodes': undirected_graph.number_of_nodes(),
                'edges': undirected_graph.number_of_edges(),
                'density': nx.density(undirected_graph),
                'is_connected': nx.is_connected(undirected_graph),
                'num_components': nx.number_connected_components(undirected_graph)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating centrality measures: {e}")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': str(e)
        }
