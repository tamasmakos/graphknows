"""Lightweight string-based coreference resolution for raw extraction data.

Public symbols:
- :func:`resolve_extraction_coreferences` — normalize entity surface forms in a
  batch of relation triplets.
- :func:`_canonicalize_entity_name`, :func:`_string_similarity`,
  :func:`_token_similarity`, :func:`_are_coreferent` — lower-level helpers.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


def _canonicalize_entity_name(name: str) -> str:
    """Return a lowercased, punctuation-stripped canonical form."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    return name


def _string_similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio for *a* and *b*."""
    return SequenceMatcher(None, a, b).ratio()


def _token_similarity(a: str, b: str) -> float:
    """Return Jaccard index over word tokens (handles word-order variation)."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _are_coreferent(a: str, b: str, threshold: float) -> bool:
    """Return True if *a* and *b* are likely the same entity."""
    if _string_similarity(a, b) >= threshold:
        return True
    if _token_similarity(a, b) >= 0.9:
        return True
    return False


def resolve_extraction_coreferences(
    relations: list[tuple[str, str, str, dict[str, Any]]],
    entities: list[str],
    similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    """Normalize entity surface forms within a single extraction batch.

    Identifies variations of the same entity name and maps them to a single
    representative form; then remaps the provided relation triplets.

    Args:
        relations: ``(head, relation, tail, props)`` tuples from one extraction pass.
        entities: Isolated entity name strings from the same pass.
        similarity_threshold: String-similarity threshold for coreference (default 0.85).

    Returns:
        ``{cleaned_relations, entity_mappings, debug_log}`` dict.
    """
    try:
        debug_log: list[str] = []

        # 1. Collect all surface forms
        originals: set[str] = set()
        for item in relations or []:
            if len(item) >= 3:
                s, _, t = item[0], item[1], item[2]
                if isinstance(s, str):
                    originals.add(s)
                if isinstance(t, str):
                    originals.add(t)
        for e in entities or []:
            if isinstance(e, str):
                originals.add(e)

        # 2. Initial canonicalization for grouping
        orig_to_canon: dict[str, str] = {o: _canonicalize_entity_name(o) for o in originals}
        canonicals: list[str] = sorted(set(orig_to_canon.values()))

        # 3. Greedy grouping by similarity
        rep_for: dict[str, str] = {}
        representatives: list[str] = []

        for c in canonicals:
            placed = False
            for r in representatives:
                if _are_coreferent(c, r, similarity_threshold):
                    best = r if len(r) >= len(c) else c
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

        # 4. Map canonical representatives back to best original surface form
        canon_to_best_original: dict[str, str] = {}
        for r in representatives:
            candidates = [o for o, c in orig_to_canon.items() if rep_for.get(c) == r]
            if candidates:
                best_orig = sorted(
                    candidates,
                    key=lambda x: (len(x), sum(1 for ch in x if ch.isupper())),
                    reverse=True,
                )[0]
                canon_to_best_original[r] = best_orig

        entity_mappings: dict[str, str] = {}
        for o, c in orig_to_canon.items():
            rep_canon = rep_for.get(c, c)
            final_name = canon_to_best_original.get(rep_canon, o)
            entity_mappings[o] = final_name

        # 5. Remap relations
        cleaned_list: list[tuple[str, str, str, dict[str, Any]]] = []
        for item in relations or []:
            if len(item) == 4:
                s, r, t, props = item
            else:
                s, r, t = item[0], item[1], item[2]
                props = {}

            cs = entity_mappings.get(s, s)
            ct = entity_mappings.get(t, t)
            if not cs or not ct or cs == ct:
                continue
            cleaned_list.append((cs, r, ct, props))

        debug_log.append(f"normalized_entities={len(entity_mappings)} reps={len(representatives)}")

        return {
            "cleaned_relations": cleaned_list,
            "entity_mappings": entity_mappings,
            "debug_log": debug_log,
        }
    except Exception as e:
        logger.warning(f"Lightweight coreference normalization failed: {e}")
        return {
            "cleaned_relations": relations,
            "entity_mappings": {},
            "debug_log": [f"error: {str(e)}"],
        }
