"""Shared RRF (Reciprocal Rank Fusion) constants and scoring used by STM and LTM recall."""

from __future__ import annotations

import math

RRF_K: int = 60

# Extra weight applied to relation-expanded traversal results in Source C.
RRF_RELATION_BOOST: float = 1.5

# Score-aware fusion: weight applied to normalised cosine similarity as a
# tiebreaker on top of rank-based RRF in the semantic collector. Small values
# break ties without overriding multi-source rank agreement.
FUSION_ALPHA: float = 0.005

# Chunk-id prefix for topic results — excluded from vector store bulk-fetch.
TOPIC_PREFIX: str = "topic:"

# ── PageRank boost ─────────────────────────────────────────────────────────
# Maximum multiplicative boost applied when an entity has high PageRank
# centrality.  Bounded to 1.0–(1.0 + PAGERANK_MAX_BOOST) so absent/zero
# pagerank is a clean no-op (multiply by 1.0).
PAGERANK_MAX_BOOST: float = 0.3

# ── IDF discriminativeness ────────────────────────────────────────────────
# Entities that appear in more than this fraction of the corpus are penalised
# by an IDF-based smooth down-weight (instead of hard-dropping them from
# traversal — the hard drop lives in _discriminative_entities).
IDF_SATURATION_RATIO: float = 0.5

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


def pagerank_boost(pagerank: float) -> float:
    """Return a bounded multiplicative boost in [1.0, 1 + PAGERANK_MAX_BOOST].

    *pagerank* is expected to be a raw PageRank value (0–1 after normalisation,
    or an unnormalised value from a graph projection).  We apply a log
    compression so highly central nodes get a meaningful but bounded lift.

    Args:
        pagerank: PageRank centrality score.  Missing/zero → 0 → no boost.

    Returns:
        Multiplier in [1.0, 1.0 + PAGERANK_MAX_BOOST].
    """
    if not pagerank or pagerank <= 0.0:
        return 1.0
    # log(1 + x) grows fast for small x and saturates for large x.
    # Clamp the raw value to [0, 1] before the log to keep the output bounded.
    clamped = min(float(pagerank), 1.0)
    raw_boost = math.log1p(clamped) / math.log1p(1.0)  # normalise to [0, 1]
    return 1.0 + raw_boost * PAGERANK_MAX_BOOST


def idf_weight(doc_freq: int, corpus_size: int) -> float:
    """Return a smooth IDF down-weight for an entity based on its corpus frequency.

    Entities that appear in more than ``IDF_SATURATION_RATIO`` of the corpus
    are ubiquitous (conversation participants, stop-concepts) and receive a
    weight < 1.0.  Rare entities get weight = 1.0.

    Args:
        doc_freq: Number of chunks the entity appears in.
        corpus_size: Total number of chunks in the corpus.

    Returns:
        Weight in (0, 1].  Returns 1.0 when *corpus_size* is 0.
    """
    if corpus_size <= 0 or doc_freq <= 0:
        return 1.0
    ratio = doc_freq / corpus_size
    if ratio <= IDF_SATURATION_RATIO:
        return 1.0
    # Smooth penalty: weight = IDF_SATURATION_RATIO / ratio
    # At ratio=1.0 → weight = IDF_SATURATION_RATIO (e.g. 0.5)
    # At ratio=IDF_SATURATION_RATIO → weight = 1.0 (boundary, no penalty)
    return IDF_SATURATION_RATIO / ratio
