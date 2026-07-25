"""Shared RRF (Reciprocal Rank Fusion) constants and scoring used by STM and LTM recall."""

from __future__ import annotations

RRF_K: int = 60

# Extra weight applied to relation-expanded traversal results in Source C.
RRF_RELATION_BOOST: float = 1.5

# Score-aware fusion: weight applied to normalised cosine similarity as a
# tiebreaker on top of rank-based RRF in the semantic collector. Small values
# break ties without overriding multi-source rank agreement.
FUSION_ALPHA: float = 0.005


def rrf_score(rank: int, boost: float = 1.0) -> float:
    """Return the RRF contribution for a result at *rank* (0-based)."""
    return boost / (RRF_K + rank + 1)
