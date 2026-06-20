"""Shared RRF (Reciprocal Rank Fusion) constants and scoring used by STM and LTM recall."""

from __future__ import annotations

RRF_K: int = 60

# Extra weight applied to relation-expanded traversal results in Source C.
RRF_RELATION_BOOST: float = 1.5

# Chunk-id prefix for topic results — excluded from vector store bulk-fetch.
TOPIC_PREFIX: str = "topic:"

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
    return score * _SHAPE_FACTORS.get(memory_class, 1.0)
