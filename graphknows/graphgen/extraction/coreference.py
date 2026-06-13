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

log = logging.getLogger(__name__)


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


def _collect_surface_forms(
    relations: list[tuple[str, str, str, dict[str, Any]]],
    entities: list[str],
) -> set[str]:
    """Collect all unique entity surface forms from relations and entity list."""
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
    return originals


def _build_canonical_groups(
    originals: set[str],
    similarity_threshold: float,
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Canonicalize surface forms and group coreferent variants.

    Returns (orig_to_canon, representatives, rep_for).
    """
    orig_to_canon: dict[str, str] = {o: _canonicalize_entity_name(o) for o in originals}
    canonicals: list[str] = sorted(set(orig_to_canon.values()))

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

    return orig_to_canon, representatives, rep_for


def _best_original_per_rep(
    representatives: list[str],
    orig_to_canon: dict[str, str],
    rep_for: dict[str, str],
) -> dict[str, str]:
    """Map each canonical representative back to the best original surface form."""
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
    return canon_to_best_original


def _remap_relations(
    relations: list[tuple[str, str, str, dict[str, Any]]],
    entity_mappings: dict[str, str],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Apply entity_mappings to relation triplets, dropping self-loops."""
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
    return cleaned_list


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
        originals = _collect_surface_forms(relations, entities)
        orig_to_canon, representatives, rep_for = _build_canonical_groups(
            originals, similarity_threshold
        )
        canon_to_best_original = _best_original_per_rep(representatives, orig_to_canon, rep_for)

        entity_mappings: dict[str, str] = {}
        for o, c in orig_to_canon.items():
            rep_canon = rep_for.get(c, c)
            final_name = canon_to_best_original.get(rep_canon, o)
            entity_mappings[o] = final_name

        cleaned_list = _remap_relations(relations, entity_mappings)
        debug_log = [f"normalized_entities={len(entity_mappings)} reps={len(representatives)}"]

        return {
            "cleaned_relations": cleaned_list,
            "entity_mappings": entity_mappings,
            "debug_log": debug_log,
        }
    except Exception as e:
        log.warning("Lightweight coreference normalization failed: %s", e)
        return {
            "cleaned_relations": relations,
            "entity_mappings": {},
            "debug_log": [f"error: {str(e)}"],
        }
