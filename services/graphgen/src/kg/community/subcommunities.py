import networkx as nx
from collections import defaultdict
from typing import Dict, List
import logging

import networkx as nx
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def add_enhanced_community_attributes_to_graph(graph: nx.DiGraph, communities: Dict[str, int], 
                                             subcommunities: Dict[str, Tuple[int, int]]) -> nx.DiGraph:
    """Enhanced version that creates proper hierarchical connections: Entities→Subtopics→Parent Topics
    Expects subcommunities mapping: entity_id -> (parent_community_id, local_sub_id).
    """
    logger.info("Creating PROPER topic hierarchy: Entities→Subtopics→Parent Topics...")
    
    # 0. CLEANUP: Remove existing TOPIC and SUBTOPIC nodes to prevent duplication
    # Since we are regenerating communities globally, old topics are obsolete.
    nodes_to_remove = []
    for n, data in graph.nodes(data=True):
        if data.get('node_type') in ['TOPIC', 'SUBTOPIC']:
            nodes_to_remove.append(n)
    
    if nodes_to_remove:
        logger.info(f"Removing {len(nodes_to_remove)} existing topic/subtopic nodes before regeneration...")
        graph.remove_nodes_from(nodes_to_remove)

    # Create ParentTopic nodes  
    topic_nodes_created = 0
    unique_communities = set(communities.values())
    for comm_id in unique_communities:
        topic_node_id = f"TOPIC_{comm_id}"
        if topic_node_id not in graph:
            graph.add_node(topic_node_id,
                          id=topic_node_id,
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
    
    # Track which entities have been assigned to a topic node
    entities_with_topic = set()
    
    for node_id, pair in subcommunities.items():
        if node_id not in graph:
            continue
        # Only connect ENTITY_CONCEPT nodes to topics
        node_data = graph.nodes.get(node_id, {})
        if node_data.get('node_type') not in ['ENTITY_CONCEPT', 'PLACE', 'ENTITY']:
            continue
            
        parent_comm_id, local_sub_id = pair
        topic_node_id = f"TOPIC_{parent_comm_id}"
        
        # If we have a local sub_id, use a Subtopic node
        if local_sub_id is not None and local_sub_id != -1:
            sub_node_id = f"SUBTOPIC_{parent_comm_id}_{local_sub_id}"
            if sub_node_id not in graph:
                graph.add_node(sub_node_id,
                              id=sub_node_id,
                              node_type="SUBTOPIC",
                              graph_type="topic",
                              community_id=parent_comm_id,
                              subtopic_local_id=local_sub_id,
                              name=f"Subtopic {parent_comm_id}-{local_sub_id}")
                subtopic_nodes_created += 1
            
            # Entity -> Subtopic
            if not graph.has_edge(node_id, sub_node_id):
                graph.add_edge(node_id, sub_node_id,
                              label="IN_TOPIC",
                              graph_type="topic")
                in_topic_edges_created += 1
                
            # Subtopic -> Parent topic
            if not graph.has_edge(sub_node_id, topic_node_id):
                graph.add_edge(sub_node_id, topic_node_id,
                              label="PARENT_TOPIC",
                              graph_type="topic")
                parent_topic_edges_created += 1
        else:
            # Connect Entity -> Topic directly if no subtopic
            if not graph.has_edge(node_id, topic_node_id):
                graph.add_edge(node_id, topic_node_id,
                              label="IN_TOPIC",
                              graph_type="topic")
                in_topic_edges_created += 1
        
        entities_with_topic.add(node_id)
        
    # FALLBACK: Ensure EVERY entity assigned to a community is connected to its TOPIC
    # even if it wasn't in the subcommunities mapping (e.g. communities with 1 node)
    for node_id, comm_id in communities.items():
        if node_id in entities_with_topic or node_id not in graph:
            continue
            
        node_data = graph.nodes.get(node_id, {})
        if node_data.get('node_type') not in ['ENTITY_CONCEPT', 'PLACE', 'ENTITY']:
            continue
            
        topic_node_id = f"TOPIC_{comm_id}"
        if topic_node_id not in graph:
            # Should already exist, but just in case
            graph.add_node(topic_node_id, id=topic_node_id, node_type="TOPIC", 
                          graph_type="topic", community_id=comm_id, name=f"Topic {comm_id}")
            
        if not graph.has_edge(node_id, topic_node_id):
            graph.add_edge(node_id, topic_node_id,
                          label="IN_TOPIC",
                          graph_type="topic")
            in_topic_edges_created += 1
    
    logger.info(f"Created {subtopic_nodes_created} Subtopic nodes")
    logger.info(f"Created {topic_nodes_created} Topic nodes") 
    logger.info(f"Created {in_topic_edges_created} entity→subtopic IN_TOPIC relationships")
    logger.info(f"Created {parent_topic_edges_created} subtopic→parent PARENT_TOPIC relationships")
    logger.info("✅ PROPER HIERARCHY: Entities→Subtopics→Parent Topics")
    
    return graph
