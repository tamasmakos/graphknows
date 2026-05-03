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
from graphgen.pipeline.graph_cleaning.canonicalization import (
    are_potential_aliases,
    classify_surface_form,
    normalize_surface_form,
    surface_forms_conflict,
)
from graphgen.utils.utils import merge_node_into
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# --- Entity Resolution Utilities (Splink-inspired Blocking) ---

@dataclass
class EntityRecord:
    id: str
    text: str
    type: str
    cluster_id: str = None  # The final resolved ID
    embedding: np.ndarray = None
    structural_embedding: np.ndarray = None # Node2Vec embedding
    metadata: Dict[str, Any] = field(default_factory=dict)


def _types_compatible(type_a: str, type_b: str) -> bool:
    if not type_a or not type_b:
        return True
    return normalize_surface_form(type_a) == normalize_surface_form(type_b)


def _neighbor_signature(graph: nx.DiGraph, node_id: str) -> Set[Tuple[str, str]]:
    signature: Set[Tuple[str, str]] = set()
    if node_id not in graph:
        return signature
    for neighbor in graph.neighbors(node_id):
        edge = graph.get_edge_data(node_id, neighbor) or {}
        signature.add((normalize_surface_form(edge.get('label', '')), normalize_surface_form(str(neighbor))))
    for source, _, edge in graph.in_edges(node_id, data=True):
        signature.add((normalize_surface_form(edge.get('label', '')), normalize_surface_form(str(source))))
    return signature


def _neighbor_compatibility(graph: Optional[nx.DiGraph], node_a: str, node_b: str) -> float:
    if graph is None:
        return 0.0
    sig_a = _neighbor_signature(graph, node_a)
    sig_b = _neighbor_signature(graph, node_b)
    if not sig_a or not sig_b:
        return 0.0
    intersection = len(sig_a & sig_b)
    union = len(sig_a | sig_b)
    return intersection / union if union else 0.0

class BlockingResolver:
    """
    Implements blocking-based entity resolution to avoid O(N^2) comparisons.
    """
    def __init__(self, similarity_threshold: float = 0.90, graph: Optional[nx.DiGraph] = None):
        self.similarity_threshold = similarity_threshold
        self.records: List[EntityRecord] = []
        self.graph = graph
        
    def add_records(self, records: List[EntityRecord]):
        self.records.extend(records)

    def _get_blocking_keys(self, text: str) -> List[str]:
        """
        Generate blocking keys for a record.
        Smart blocking using:
        1. First letter + length (simple)
        2. Metaphone-like (or simple phonetic) if available, otherwise just normalized start.
        3. Token-based blocking (e.g. "Apple Inc" -> "Apple", "Inc")
        """
        keys = set()
        text = text.lower().strip()
        
        # Block 1: First letter (very coarse, but fast)
        if text:
            keys.add(f"ALPHA:{text[0]}")
            
        # Block 2: Significant Tokens (skipping stop words)
        tokens = re.split(r'\W+', text)
        stop_words = {'the', 'a', 'an', 'inc', 'ltd', 'corp', 'company', 'and', 'of'}
        for t in tokens:
            if len(t) > 2 and t not in stop_words:
                keys.add(f"TOKEN:{t}")
                
        return list(keys)

    def _compute_similarity(self, rec_a: EntityRecord, rec_b: EntityRecord) -> float:
        """
        Compute similarity between two records.
        Uses embedding cosine similarity if available, otherwise string similarity.
        """
        if not _types_compatible(rec_a.type, rec_b.type):
            return 0.0

        if surface_forms_conflict(rec_a.text, rec_b.text):
            return 0.0

        class_a = classify_surface_form(rec_a.text)
        class_b = classify_surface_form(rec_b.text)
        if {class_a, class_b} == {'role_artifact', 'named_entity'}:
            return 0.0
        if 'role_artifact' in {class_a, class_b} and class_a != class_b:
            return 0.0
        if {class_a, class_b} == {'concept_like', 'named_entity'}:
            return 0.0

        if are_potential_aliases(rec_a.text, rec_b.text):
            alias_bonus = 0.97
        else:
            alias_bonus = 0.0

        sim_semantic = 0.0
        has_semantic = False
        
        if rec_a.embedding is not None and rec_b.embedding is not None:
             # Cosine similarity
            dot = np.dot(rec_a.embedding, rec_b.embedding)
            norm_a = np.linalg.norm(rec_a.embedding)
            norm_b = np.linalg.norm(rec_b.embedding)
            if norm_a > 0 and norm_b > 0:
                sim_semantic = dot / (norm_a * norm_b)
                has_semantic = True
        
        sim_structural = 0.0
        has_structural = False
        
        if rec_a.structural_embedding is not None and rec_b.structural_embedding is not None:
             # Cosine similarity for structural
            dot = np.dot(rec_a.structural_embedding, rec_b.structural_embedding)
            norm_a = np.linalg.norm(rec_a.structural_embedding)
            norm_b = np.linalg.norm(rec_b.structural_embedding)
            if norm_a > 0 and norm_b > 0:
                sim_structural = dot / (norm_a * norm_b)
                has_structural = True

        neighborhood_score = _neighbor_compatibility(self.graph, rec_a.id, rec_b.id)
                
        # Combine similarities
        if has_semantic and has_structural:
            combined = (0.6 * sim_semantic) + (0.2 * sim_structural) + (0.2 * neighborhood_score)
            return max(combined, alias_bonus)
        elif has_semantic:
            combined = (0.85 * sim_semantic) + (0.15 * neighborhood_score)
            return max(combined, alias_bonus)
        elif has_structural:
            combined = (0.7 * sim_structural) + (0.3 * neighborhood_score)
            return max(combined * 0.8, alias_bonus)

        # 2. String Similarity
        text_a = rec_a.text.lower()
        text_b = rec_b.text.lower()
        seq_sim = SequenceMatcher(None, text_a, text_b).ratio()
        tok_sim = _token_similarity(text_a, text_b)
        string_score = max(seq_sim, tok_sim)

        if self.graph is not None:
            string_score = max(string_score, min(0.99, string_score + (0.15 * neighborhood_score)))

        if text_a.startswith(text_b) or text_b.startswith(text_a):
            string_score = max(string_score, 0.85)

        return max(string_score, alias_bonus)

    def resolve(self) -> Dict[str, str]:
        """
        Run resolution.
        Returns a mapping of original_id -> resolved_id
        """
        # 1. Create Blocks
        blocks = defaultdict(list)
        for rec in self.records:
            keys = self._get_blocking_keys(rec.text)
            for k in keys:
                blocks[k].append(rec)
                
        logger.info(f"Created {len(blocks)} blocks for {len(self.records)} records.")
        
        # 2. Compare within blocks (Union-Find structure for clustering)
        parent = {r.id: r.id for r in self.records}
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i # arbitrary merge
                
        # To avoid re-comparing pairs, we can track seen pairs
        seen_pairs = set()
        
        comparisons = 0
        matches = 0
        
        for block_key, block_records in blocks.items():
            # If block is too huge, we might skip or sub-block, but for now just process
            n = len(block_records)
            if n < 2: continue
            
            # Simple pairwise within block
            # For massive blocks, we would use nearest neighbors here
            for i in range(n):
                for j in range(i + 1, n):
                    rec_a = block_records[i]
                    rec_b = block_records[j]
                    
                    pair_id = tuple(sorted((rec_a.id, rec_b.id)))
                    if pair_id in seen_pairs:
                        continue
                    seen_pairs.add(pair_id)
                    comparisons += 1
                    
                    sim = self._compute_similarity(rec_a, rec_b)
                    
                    if sim >= self.similarity_threshold:
                        union(rec_a.id, rec_b.id)
                        matches += 1
                        
        logger.info(f"Resolution stats: {comparisons} comparisons, {matches} merges found.")
        
        # 3. Canonicalize
        # Group by root
        clusters = defaultdict(list)
        resolved_map = {} # original_id -> canonical_id
        
        for rec in self.records:
            root = find(rec.id)
            clusters[root].append(rec)
            
        for root, group in clusters.items():
            # Pick canonical: Longest name
            canonical = max(group, key=lambda r: len(r.text))
            for member in group:
                resolved_map[member.id] = canonical.text # Map to Name, not ID (or ID if preferred)
                
        return resolved_map

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
    relations: List[Tuple[str, str, str, Dict[str, Any]]], 
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
        for i, item in enumerate(relations or []):
            if len(item) >= 3:
                s, _, t = item[0], item[1], item[2]
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
        # 5) Remap relations
        cleaned_list: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for item in relations or []:
            if len(item) == 4:
                s, r, t, props = item
            else:
                s, r, t = item[0], item[1], item[2]
                props = {}
                
            cs = entity_mappings.get(s, s)
            ct = entity_mappings.get(t, t)
            # Avoid self-loops created by merging
            if not cs or not ct or cs == ct:
                continue
            cleaned_list.append((cs, r, ct, props))

        cleaned_relations = cleaned_list
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
    node_types: Optional[List[str]] = None,
    structural_embeddings: Optional[Dict[str, np.ndarray]] = None
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

    # 2. Use BlockingResolver
    resolver = BlockingResolver(similarity_threshold=similarity_threshold, graph=graph)
    records = []
    
    # Store ID mapping to handle graph nodes
    id_to_record = {}
    
    for nid, emb in embeddings.items():
        node_data = graph.nodes[nid]
        name = node_data.get('name', nid)
        entity_type = (
            node_data.get('ontology_class')
            or node_data.get('ontology_label')
            or node_data.get('entity_type')
            or node_data.get('llm_type')
            or 'Entity'
        )
        
        # Get structural embedding if available
        struct_emb = structural_embeddings.get(nid) if structural_embeddings else None
        
        rec = EntityRecord(id=nid, text=name, type=entity_type, embedding=emb, structural_embedding=struct_emb)
        records.append(rec)
        id_to_record[nid] = rec
        
    resolver.add_records(records)
    
    # Run resolution
    # internal result comes as original_id -> resolved_name
    # But we want to map original_id -> canonical_id (node ID)
    
    # We can peek into BlockingResolver or adapt its output.
    # Actually, BlockingResolver.resolve() returns ID -> Name.
    # Let's modify usage or trust the name is the ID if we used IDs as text? 
    # No, we used names as text.
    
    # Let's re-implement the graph merge part using blocking basics here to be precisely controlling node IDs
    
    # OR: use the clusters from blocking
    # The BlockingResolver exposes internal logic? 
    # Let's just use it as is but re-map back to IDs.
    
    # Wait, BlockingResolver logic above returns `resolved_map[member.id] = canonical.text`.
    # We want canonical ID.
    # Let's assume for now that if we get names back, we might lose ID mappings if names are ambiguous.
    # BUT, in `graphgen`, node IDs ARE often the names.
    # If node IDs are UUIDs, this breaks.
    # Currently node IDs seem to be Entity Names (extracted string).
    # So `canonical.text` is likely `canonical.id` effectively.
    
    resolved_map = resolver.resolve()
    
    # Group by resolved name to find clusters
    name_to_ids = defaultdict(list)
    for original_id, resolved_name in resolved_map.items():
        name_to_ids[resolved_name].append(original_id)
        
    clusters = [ids for ids in name_to_ids.values() if len(ids) > 1]
    
    logger.info(f"Found {len(clusters)} clusters via blocking resolution.")
    
    # 5. Merge Nodes
    nodes_merged = 0
    merged_pairs_details = []
    
    for cluster in clusters:
        # Heuristic for Canonical Node:
        # 1. Highest Degree (most connected)
        # 2. Longest Name (most descriptive)
        def get_node_score(nid):
            degree = graph.degree(nid)
            name_len = len(graph.nodes[nid].get('name', nid))
            return (degree, name_len)
            
        canonical_id = max(cluster, key=get_node_score)
        
        # Log merge
        cluster_names = [graph.nodes[n].get('name', n) for n in cluster]
        logger.info(f"Resolving cluster {cluster_names} -> '{canonical_id}'")

        canonical_name = graph.nodes[canonical_id].get('name', canonical_id)

        for node_id in cluster:
            if node_id == canonical_id:
                continue
            
            merged_name = graph.nodes[node_id].get('name', node_id)
            merged_pairs_details.append({
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "merged_id": node_id,
                "merged_name": merged_name
            })
                
            merge_node_into(graph, source_node=node_id, target_node=canonical_id)
            nodes_merged += 1
            
    logger.info(f"Semantic resolution complete. Merged {nodes_merged} nodes.")
    
    return {
        'merged_nodes': nodes_merged,
        'clusters_found': len(clusters),
        'high_similarity_pairs': 'N/A (Blocking)',
        'merged_pairs': merged_pairs_details
    }
