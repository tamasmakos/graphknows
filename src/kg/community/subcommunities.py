import networkx as nx
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Dict, List
import logging

import networkx as nx
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def split_community_with_graph_clustering(subgraph: nx.Graph, num_clusters: int, 
                                        node_list: List[str]) -> Dict[str, int]:
    """Split a community using graph-aware clustering"""
    
    if subgraph.number_of_nodes() < num_clusters:
        # More clusters than nodes - each node gets its own cluster
        return {node: i for i, node in enumerate(node_list)}
    
    if subgraph.number_of_edges() == 0:
        # No connections - simple round-robin assignment
        return {node: i % num_clusters for i, node in enumerate(node_list)}
    
    try:
        # Use spectral clustering based on graph structure
        adjacency = nx.adjacency_matrix(subgraph)
        
        # Convert to dense for small graphs
        if adjacency.shape[0] < 1000:
            adjacency = adjacency.todense()
        
        # Simple k-means on adjacency matrix rows
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(adjacency)
        
        node_to_cluster = {}
        for i, node in enumerate(subgraph.nodes()):
            node_to_cluster[node] = cluster_labels[i]
        
        return node_to_cluster
        
    except Exception as e:
        logger.warning(f"Spectral clustering failed: {e}, using simple assignment")
        # Fallback to simple assignment
        return {node: i % num_clusters for i, node in enumerate(node_list)}



def add_enhanced_community_attributes_to_graph(graph: nx.DiGraph, communities: Dict[str, int], 
                                             subcommunities: Dict[str, Tuple[int, int]]) -> nx.DiGraph:
    """Enhanced version that creates proper hierarchical connections: Entities→Subtopics→Parent Topics
    Expects subcommunities mapping: entity_id -> (parent_community_id, local_sub_id).
    """
    logger.info("Creating PROPER topic hierarchy: Entities→Subtopics→Parent Topics...")
    
    # Create ParentTopic nodes  
    topic_nodes_created = 0
    unique_communities = set(communities.values())
    for comm_id in unique_communities:
        topic_node_id = f"TOPIC_{comm_id}"
        if topic_node_id not in graph:
            graph.add_node(topic_node_id,
                          node_type="TOPIC", 
                          graph_type="topic",
                          community_id=comm_id,
                          name=f"Topic {comm_id}")  # Temporary name until summarization sets title
            topic_nodes_created += 1
    
    # Create Subtopic nodes and edges
    subtopic_nodes_created = 0
    in_topic_edges_created = 0
    parent_topic_edges_created = 0
    created_sub_nodes = set()
    
    for node_id, pair in subcommunities.items():
        if node_id not in graph:
            continue
        # Only connect ENTITY_CONCEPT nodes to topics
        node_data = graph.nodes.get(node_id, {})
        if node_data.get('node_type') != 'ENTITY_CONCEPT':
            continue
        parent_comm_id, local_sub_id = pair
        sub_node_id = f"SUBTOPIC_{parent_comm_id}_{local_sub_id}"
        if sub_node_id not in graph:
            graph.add_node(sub_node_id,
                          node_type="SUBTOPIC",
                          graph_type="topic",
                          community_id=parent_comm_id,
                          subtopic_local_id=local_sub_id,
                          name=f"Subtopic {parent_comm_id}-{local_sub_id}")  # Temporary name until summarization sets title
            subtopic_nodes_created += 1
        created_sub_nodes.add(sub_node_id)
        # Entity -> Subtopic
        if not graph.has_edge(node_id, sub_node_id):
            graph.add_edge(node_id, sub_node_id,
                          label="IN_TOPIC",
                          graph_type="topic")
            in_topic_edges_created += 1
        # Subtopic -> Parent topic
        topic_node_id = f"TOPIC_{parent_comm_id}"
        if not graph.has_edge(sub_node_id, topic_node_id):
            graph.add_edge(sub_node_id, topic_node_id,
                          label="PARENT_TOPIC",
                          graph_type="topic")
            parent_topic_edges_created += 1
    
    logger.info(f"Created {subtopic_nodes_created} Subtopic nodes")
    logger.info(f"Created {topic_nodes_created} Topic nodes") 
    logger.info(f"Created {in_topic_edges_created} entity→subtopic IN_TOPIC relationships")
    logger.info(f"Created {parent_topic_edges_created} subtopic→parent PARENT_TOPIC relationships")
    logger.info("✅ PROPER HIERARCHY: Entities→Subtopics→Parent Topics")
    
    return graph
