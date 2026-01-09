import os
import logging
import json
import networkx as nx
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_community_summary_comparison(graph: nx.DiGraph, output_dir: str) -> Dict[str, Any]:
    """
    Generate a comparison of topic summaries and save to file.
    """
    logger.info("Generating topic summary comparison...")
    
    comparison = {
        "topics": [],
        "subtopics": [],
        "generated_at": datetime.now().isoformat()
    }
    
    # Collect topic data
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'TOPIC':
            topic_data = {
                "id": node,
                "title": data.get('title', 'Untitled'),
                "summary": data.get('summary', 'No summary'),
                "entity_count": 0, # To be filled
                "subtopic_count": 0 # To be filled
            }
            
            # Count entities
            entities = [p for p in graph.predecessors(node) if graph.nodes[p].get('node_type') == 'ENTITY_CONCEPT']
            topic_data["entity_count"] = len(entities)
            
            # Count subtopics
            subtopics = [s for s in graph.successors(node) if graph.nodes[s].get('node_type') == 'SUBTOPIC']
            topic_data["subtopic_count"] = len(subtopics)
            
            comparison["topics"].append(topic_data)
            
        elif data.get('node_type') == 'SUBTOPIC':
            sub_data = {
                "id": node,
                "title": data.get('title', 'Untitled'),
                "summary": data.get('summary', 'No summary'),
                "entity_count": 0,
                "parent_topic": None
            }
            
            # Count entities
            entities = [p for p in graph.predecessors(node) if graph.nodes[p].get('node_type') == 'ENTITY_CONCEPT']
            sub_data["entity_count"] = len(entities)
            
            # Find parent
            parents = [p for p in graph.predecessors(node) if graph.nodes[p].get('node_type') == 'TOPIC']
            if parents:
                sub_data["parent_topic"] = parents[0]
                
            comparison["subtopics"].append(sub_data)
    
    # Save to file
    output_file = os.path.join(output_dir, "topic_summary_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
        
    logger.info(f"Topic summary comparison saved to {output_file}")
    return comparison


