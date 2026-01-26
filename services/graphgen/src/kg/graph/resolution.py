"""
Unified Entity Resolution Module.

This module handles:
1. Fast string-based coreference resolution (for initial extraction).
2. Semantic entity resolution using vector embeddings (for graph refinement).
3. Merging of identical entities to keep the graph clean.
"""

import logging
import re
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from difflib import SequenceMatcher
from kg.graph.utils import merge_node_into

logger = logging.getLogger(__name__)

# --- Part 1: String-Based Helpers (formerly coref.py) ---

def _canonicalize_entity_name(name: str) -> str:
    """
    Canonicalize entity name:
    - Lowercase
    - Remove punctuation
    - Singularize (simple heuristic)
    """
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    # Simple singularization (can be improved)
    # if name.endswith('s') and not name.endswith('ss'):
    #     name = name[:-1]
    return name

def _string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()

def _token_similarity(a: str, b: str) -> float:
    """
    Calculate token-based similarity (Jaccard index).
    Handles word reordering (e.g. "President of ECB" vs "ECB President").
    """
    set_a = set(a.split())
    set_b = set(b.split())
    
    if not set_a or not set_b:
        return 0.0
        
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0

def _are_coreferent(a: str, b: str, threshold: float) -> bool:
    """
    Check if two strings are likely coreferent.
    """
    # 1. Direct string similarity
    if _string_similarity(a, b) >= threshold:
        return True
        
    # 2. Token-based similarity (slightly higher threshold)
    if _token_similarity(a, b) >= 0.9: 
        return True
        
    return False

def resolve_extraction_coreferences(
    relations: List[Tuple[str, str, str]], 
    entities: List[str],
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Lightweight entity normalization for raw extraction data.
    
    Identifies variations of the same name within a single extraction batch
    and maps them to a single representative.
    
    Args:
        relations: List of (head, relation, tail) triplets
        entities: List of isolated entity names
        similarity_threshold: Threshold for string matching (default 0.85)

    Returns:
        Dictionary containing cleaned relations and entity mappings.
    """
    try:
        debug_log: List[str] = []

        # 1) Collect all surface forms
        originals: Set[str] = set()
        for s, _, t in relations or []:
            if isinstance(s, str): originals.add(s)
            if isinstance(t, str): originals.add(t)
        for e in entities or []:
            if isinstance(e, str): originals.add(e)

        # 2) Initial canonicalization for grouping
        orig_to_canon: Dict[str, str] = {o: _canonicalize_entity_name(o) for o in originals}
        canonicals: List[str] = sorted(set(orig_to_canon.values()))

        # 3) Greedy grouping by similarity
        # rep_for maps: canonical_string -> representative_canonical_string
        rep_for: Dict[str, str] = {}
        representatives: List[str] = []
        
        for c in canonicals:
            placed = False
            for r in representatives:
                if _are_coreferent(c, r, similarity_threshold):
                    # Choose longer string as representative (usually more specific)
                    best = r if len(r) >= len(c) else c
                    
                    # If representative changes, update everything pointing to old r
                    if best != r:
                        for k, v in list(rep_for.items()):
                            if v == r:
                                rep_for[k] = best
                        representatives[representatives.index(r)] = best
                        
                    rep_for[c] = representatives[representatives.index(best)]
                    placed = True
                    break
            
            if not placed:
                representatives.append(c)
                rep_for[c] = c

        # 4) Final mapping: Original Name -> Final Representative Name
        # We need to map back to one of the Original Names that corresponds to the Representative
        # Find best original string for each canonical representative
        canon_to_best_original = {}
        for r in representatives:
            # Find all originals that map to this canonical rep
            candidates = [o for o, c in orig_to_canon.items() if rep_for.get(c) == r]
            if candidates:
                # Pick the longest/most capitalized one as the "Display Name"
                # Heuristic: longest string, then most capital letters
                best_orig = sorted(candidates, key=lambda x: (len(x), sum(1 for c in x if c.isupper())), reverse=True)[0]
                canon_to_best_original[r] = best_orig

        entity_mappings: Dict[str, str] = {}
        for o, c in orig_to_canon.items():
            rep_canon = rep_for.get(c, c)
            final_name = canon_to_best_original.get(rep_canon, o)
            entity_mappings[o] = final_name

        # 5) Remap relations
        cleaned_set: Set[Tuple[str, str, str]] = set()
        for s, r, t in relations or []:
            cs = entity_mappings.get(s, s)
            ct = entity_mappings.get(t, t)
            # Avoid self-loops created by merging
            if not cs or not ct or cs == ct:
                continue
            cleaned_set.add((cs, r, ct))

        cleaned_relations = list(cleaned_set)
        debug_log.append(f"normalized_entities={len(entity_mappings)} reps={len(representatives)}")

        return {
            'cleaned_relations': cleaned_relations,
            'entity_mappings': entity_mappings,
            'debug_log': debug_log,
        }
    except Exception as e:
        logger.warning(f"Lightweight coreference normalization failed: {e}")
        return {
            'cleaned_relations': relations,
            'entity_mappings': {},
            'debug_log': [f"error: {str(e)}"]
        }


# --- Part 2: Embedding-Based Logic (formerly similarity.py + resolution.py) ---

def _compute_similarity_matrix(embeddings: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """
    Compute pairwise cosine similarity matrix.
    """
    node_ids = list(embeddings.keys())
    n = len(node_ids)
    
    if n == 0:
        return [], np.array([])
    
    # Stack embeddings
    embedding_matrix = np.array([embeddings[nid] for nid in node_ids])
    
    # Normalize
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embedding_matrix / norms
    
    # Dot product
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return node_ids, similarity_matrix

def resolve_entities_semantically(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.95,
    node_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Identify and MERGE nodes that have very high embedding similarity.
    Does NOT create edges. Directly merges nodes.
    
    Args:
        graph: The Knowledge Graph (modified in-place).
        similarity_threshold: Threshold for considering entities identical (default 0.95).
        node_types: List of node types to consider (default: ['ENTITY_CONCEPT']).
        
    Returns:
        Statistics about the merge operation.
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT']
        
    logger.info(f"Starting semantic entity resolution (threshold={similarity_threshold})...")
    
    # 1. Collect embeddings
    embeddings: Dict[str, np.ndarray] = {}
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') in node_types:
            emb = node_data.get('embedding')
            if emb is not None:
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings[node_id] = emb

    if len(embeddings) < 2:
        return {'merged_nodes': 0, 'clusters_found': 0}

    # 2. Compute Matrix
    node_ids, similarity_matrix = _compute_similarity_matrix(embeddings)
    n = len(node_ids)
    
    # 3. Build temporary graph of "identical" nodes
    # We use a temporary graph to find connected components (clusters)
    sim_graph = nx.Graph()
    sim_graph.add_nodes_from(node_ids)
    
    high_sim_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                sim_graph.add_edge(node_ids[i], node_ids[j])
                high_sim_pairs += 1

    # 4. Find Clusters
    components = list(nx.connected_components(sim_graph))
    clusters = [list(c) for c in components if len(c) > 1]
    
    logger.info(f"Found {len(clusters)} clusters of identical entities (from {high_sim_pairs} pairs).")
    
    # 5. Merge Nodes
    nodes_merged = 0
    
    for cluster in clusters:
        # Heuristic for Canonical Node:
        # 1. Highest Degree (most connected)
        # 2. Longest Name (most descriptive)
        def get_node_score(nid):
            degree = graph.degree(nid)
            # Prefer real names over IDs if possible, though nid usually is the name
            name_len = len(graph.nodes[nid].get('name', nid))
            return (degree, name_len)
            
        canonical_id = max(cluster, key=get_node_score)
        
        # Log merge
        cluster_names = [graph.nodes[n].get('name', n) for n in cluster]
        logger.info(f"Resolving cluster {cluster_names} -> '{canonical_id}'")

        for node_id in cluster:
            if node_id == canonical_id:
                continue
                
            merge_node_into(graph, source_node=node_id, target_node=canonical_id)
            nodes_merged += 1
            
    logger.info(f"Semantic resolution complete. Merged {nodes_merged} nodes.")
    
    return {
        'merged_nodes': nodes_merged,
        'clusters_found': len(clusters),
        'high_similarity_pairs': high_sim_pairs
    }
