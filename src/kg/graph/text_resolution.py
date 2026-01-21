"""
Text-based Entity Resolution Module.

This module handles the resolution (merging) of valid entities and locations
in the Knowledge Graph using text similarity metrics (Jaro-Winkler).
"""

import logging
import networkx as nx
import jellyfish
from typing import Dict, List, Set, Tuple, Any, Optional

from src.kg.graph.utils import merge_node_into

logger = logging.getLogger(__name__)

def resolve_similar_entities(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.92, # Bumped default slightly for precision if using fuzzy
    node_types: Optional[List[str]] = None
) -> List[Tuple[str, str, float]]:
    """
    Identify and merge nodes that have very high text similarity in their names.
    Uses a hybrid approach of Token Sorting and Jaro-Winkler to catch word reorderings
    and typos while maintaining precision.
    
    Args:
        graph: The Knowledge Graph (modified in-place).
        similarity_threshold: Threshold for Jaro-Winkler (0.0 to 1.0).
        node_types: List of node types to consider (default: ['ENTITY_CONCEPT', 'PLACE']).
        
    Returns:
        List of merge operations performed [(source_id, target_id, score), ...]
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT', 'PLACE']
        
    logger.info(f"Starting text-based entity resolution (threshold={similarity_threshold})...")
    
    nodes_by_type: Dict[str, List[str]] = {}
    
    for node_id, node_data in graph.nodes(data=True):
        ntype = node_data.get('node_type')
        if ntype in node_types:
            if ntype not in nodes_by_type:
                nodes_by_type[ntype] = []
            nodes_by_type[ntype].append(node_id)
            
    merges_performed = []
    
    # Helper for similarity
    def calculate_hybrid_score(s1: str, s2: str) -> float:
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if not s1 or not s2: return 0.0
        if s1 == s2: return 1.0
        
        # 1. Token Sort Ratio (Handle "New York City" == "City New York")
        tokens1 = sorted(s1.split())
        tokens2 = sorted(s2.split())
        
        # Exact token match
        if tokens1 == tokens2:
            return 0.99
            
        # Jaccard Token Set for overlap
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union if union > 0 else 0
        
        # If very high token overlap, trust it
        if jaccard > 0.8:
            return 0.95 + (jaccard - 0.8) * 0.25 # Boost to > 0.95
            
        # 2. Jaro-Winkler for typos
        jw_score = jellyfish.jaro_winkler_similarity(s1, s2)
        
        # 3. Substring check (one is substring of other and long enough)
        # Danger: "Cat" in "Caterpillar" -> Bad. "University of Florida" in "The University of Florida" -> Good.
        # Only if string length is substantial
        if len(s1) > 8 and len(s2) > 8:
            if s1 in s2 or s2 in s1:
                # Boost JW score if valid substring
                jw_score = max(jw_score, 0.95)
                
        return jw_score

    for ntype, node_ids in nodes_by_type.items():
        node_ids.sort() 
        n = len(node_ids)
        if n < 2:
            continue
            
        logger.info(f"Checking {n} nodes of type {ntype} for text similarity...")
        
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(node_ids)
        
        high_sim_pairs = 0
        
        # TODO: Optimize with blocking for large N
        for i in range(n):
            id_a = node_ids[i]
            name_a = graph.nodes[id_a].get('name', '')
            
            for j in range(i + 1, n):
                id_b = node_ids[j]
                name_b = graph.nodes[id_b].get('name', '')
                
                score = calculate_hybrid_score(name_a, name_b)
                
                if score >= similarity_threshold:
                    sim_graph.add_edge(id_a, id_b, weight=score)
                    high_sim_pairs += 1

        components = list(nx.connected_components(sim_graph))
        clusters = [list(c) for c in components if len(c) > 1]
        
        for cluster in clusters:
            # Pick canonical: Prefer higher degree, then longer name (usually more descriptive)
            def get_node_score(nid):
                degree = graph.degree(nid)
                name = graph.nodes[nid].get('name', '')
                # Check formatting (Capitalized > Lowercase)
                is_capitalized = name[0].isupper() if name else False
                return (degree, is_capitalized, len(name), name)
            
            canonical_id = max(cluster, key=get_node_score)
            canonical_name = graph.nodes[canonical_id].get('name', '')
            
            for node_id in cluster:
                if node_id == canonical_id:
                    continue
                    
                node_name = graph.nodes[node_id].get('name', '')
                score = calculate_hybrid_score(canonical_name, node_name)
                
                logger.info(f"MERGING: {node_name} ({node_id}) -> {canonical_name} ({canonical_id}) [Score: {score:.4f}]")
                merges_performed.append((node_id, canonical_id, score))
                
                merge_node_into(graph, node_id, canonical_id)

    return merges_performed


