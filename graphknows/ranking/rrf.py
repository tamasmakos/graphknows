"""Shared RRF (Reciprocal Rank Fusion) constants and scoring used by STM and LTM recall."""

from __future__ import annotations

RRF_K: int = 60

# Extra weight applied to relation-expanded traversal results in Source C.
RRF_RELATION_BOOST: float = 1.5

# Score-aware fusion: weight applied to normalised cosine similarity as a
# tiebreaker on top of rank-based RRF in the semantic collector. Small values
# break ties without overriding multi-source rank agreement.
FUSION_ALPHA: float = 0.005

_SHAPE_FACTORS: dict[str | None, float] = {
    "fact": 1.5,
    "state": 1.4,
    "episode": 1.3,
    "summary": 1.2,
    "raw_evidence": 0.6,
}


def rrf_score(rank: int, boost: float = 1.0) -> float:
    """Return the RRF contribution for a result at *rank* (0-based)."""
    return boost / (RRF_K + rank + 1)


def shape_score(score: float, memory_class: str | None) -> float:
    """Scale *score* by the shaping factor for its memory class."""
    return score * _SHAPE_FACTORS.get(memory_class, 1.0)


def community_boost(weight: object) -> float:
    """Return a bounded (1.0-3.0) score multiplier from a chunk's community weight.

    Chunks in denser, more cohesive communities are boosted. Missing, zero, or
    malformed weights are a clean no-op (multiply by 1.0), so this never
    regresses when community detection did not run.
    """
    try:
        w = float(weight) if isinstance(weight, int | float | str) else 1.0
    except (TypeError, ValueError):
        w = 1.0
    return max(1.0, min(w, 3.0))
