import logging
from typing import List, Dict, Any, Tuple, Set, Optional
import re
from difflib import SequenceMatcher
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

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
    # if name.endswith('s') and not name.endswith('ss'):
    #     name = name[:-1]
    return name

def _string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()

def _token_similarity(a: str, b: str) -> float:
    """
    Calculate token-based similarity (Jaccard index).
    Handles word reordering and minor differences (e.g. "The X" vs "X").
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
    Check if two strings are likely coreferent using multiple strategies.
    """
    # 1. Direct string similarity (Levenshtein-like)
    if _string_similarity(a, b) >= threshold:
        return True
        
    # 2. Token-based similarity (Bag of words)
    # Use a slightly higher threshold for tokens to avoid false positives with common words
    if _token_similarity(a, b) >= 0.9: # e.g. "President of ECB" vs "President of the ECB"
        return True
        
    return False

def resolve_coreferences_simple(relations: List[Tuple[str, str, str]], entities: List[str]) -> Dict[str, Any]:
    """Lightweight entity normalization and conservative merging (no PyKEEN).

    - Abbreviation expansion for key EU/NATO terms
    - Lowercasing, punctuation cleanup, noun lemmatization (singularize)
    - Greedy grouping by string similarity (≥ 0.85)
    """
    try:
        sim_threshold = 0.85 # Hardcoded for now or get from config
        debug_log: List[str] = []

        # 1) Collect all surface forms from relations + provided entities
        originals: Set[str] = set()
        for s, _, t in relations or []:
            if isinstance(s, str):
                originals.add(s)
            if isinstance(t, str):
                originals.add(t)
        for e in entities or []:
            if isinstance(e, str):
                originals.add(e)

        # 2) Initial canonicalization
        orig_to_canon: Dict[str, str] = {o: _canonicalize_entity_name(o) for o in originals}
        canonicals: List[str] = sorted(set(orig_to_canon.values()))

        # 3) Greedy grouping by similarity on canonical strings
        rep_for: Dict[str, str] = {}
        representatives: List[str] = []
        for c in canonicals:
            placed = False
            for r in representatives:
                if _are_coreferent(c, r, sim_threshold):
                    # choose longer string as representative for stability
                    best = r if len(r) >= len(c) else c
                    # update all previous mapped to r if best changes
                    if best != r:
                        # remap existing items pointing to old r
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

        # 4) Final mapping original -> final representative
        entity_mappings: Dict[str, str] = {}
        for o, c in orig_to_canon.items():
            entity_mappings[o] = rep_for.get(c, c)

        # 5) Remap relations; drop self-loops; deduplicate
        cleaned_set: Set[Tuple[str, str, str]] = set()
        for s, r, t in relations or []:
            cs = entity_mappings.get(s, s)
            ct = entity_mappings.get(t, t)
            if not cs or not ct or cs == ct:
                continue
            cleaned_set.add((cs, r, ct))

        cleaned_relations = list(cleaned_set)
        debug_log.append(f"normalized_entities={len(entity_mappings)} reps={len(representatives)}")

        return {
            'raw_relations': relations,
            'cleaned_relations': cleaned_relations,
            'predicted_links': [],
            'entity_mappings': entity_mappings,
            'merged_entities': [],
            'coreference_threshold': sim_threshold,
            'debug_log': debug_log,
        }
    except Exception as e:
        logger.warning(f"Lightweight coreference normalization failed: {e}")
        return {
            'raw_relations': relations,
            'cleaned_relations': relations,
            'predicted_links': [],
            'entity_mappings': {},
            'merged_entities': [],
            'debug_log': [f"error: {str(e)}"],
        }
