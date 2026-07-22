"""Partition-quality metrics for a topic build.

The important one is :func:`newman_modularity`. ArcadeDB exposes
``algo.modularityScore``, but it is not Newman modularity and cannot be used as
a tuning objective: measured on conv-26, an all-in-one-cluster partition scores
**0.750** against the real Louvain partition's **0.247** (Newman's Q for a single
all-encompassing cluster is 0 by definition). It appears to omit the expected-edge
term, so it rewards ever-coarser partitions and any sweep optimising it converges
on one giant topic. It does separate real structure (0.247) from a random
partition (0.095), so it is reported alongside — clearly labelled — but never
optimised against.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORD_TEXT = (
    """the a an and or but if then than that this these those there here
    is are was were be been being am do does did done have has had having
    i you he she it we they me him her them my your his its our their
    of to in on at by for with from as into about over after before
    not no nor so too very can will just should now what when where who
    which how why all any both each few more most other some such only
    own same s t don yeah yes okay ok oh hey hi like really got get
    know think going go went said say says one two really much many"""
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def newman_modularity(edges: list[tuple[str, str]], part: dict[str, int]) -> float:
    """Newman modularity Q of ``part`` over an undirected edge list.

    ``Q = sum_c (e_c / m  -  (d_c / 2m)^2)`` where ``e_c`` is the edge count
    inside community ``c`` and ``d_c`` the total degree of its members. Q is 0
    for a single all-encompassing cluster and ~0 for a random partition, which is
    exactly the behaviour the native score lacks.

    Nodes missing from ``part`` are ignored, as are self-loops.
    """
    inside: Counter[int] = Counter()
    degree: Counter[int] = Counter()
    m = 0
    for src, dst in edges:
        cs, cd = part.get(src), part.get(dst)
        if cs is None or cd is None or src == dst:
            continue
        m += 1
        degree[cs] += 1
        degree[cd] += 1
        if cs == cd:
            inside[cs] += 1
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    return sum(inside[c] / m - (degree[c] / two_m) ** 2 for c in degree)


def coherence(vectors: np.ndarray) -> tuple[float, list[float]]:
    """Mean cosine of each member to the cluster centroid, and the centroid.

    Returns ``(coherence, centroid)``. A one-member cluster is trivially
    coherent (1.0), which is why ``min_size`` exists.
    """
    if vectors.size == 0:
        return 0.0, []
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return 0.0, []
    centroid = centroid / norm
    rows = np.linalg.norm(vectors, axis=1, keepdims=True)
    rows[rows == 0.0] = 1.0
    sims = (vectors / rows) @ centroid
    return float(np.mean(sims)), centroid.tolist()


def keywords(
    member_texts: list[str], background: list[set[str]], top_k: int = 8
) -> list[str]:
    """Distinctive terms for one topic: TF-IDF against the other topics.

    A term scores highly when it is frequent *here* and rare *elsewhere*, which
    is what makes this more diagnostic than a generated sentence — it says what
    separates this topic from its neighbours rather than what it contains.
    """
    tokens = _tokenize(" ".join(member_texts))
    if not tokens:
        return []
    tf = Counter(tokens)
    n_docs = len(background) + 1
    scores: dict[str, float] = {}
    for term, count in tf.items():
        containing = 1 + sum(1 for doc in background if term in doc)
        idf = math.log(n_docs / containing)
        if idf > 0:
            scores[term] = (count / len(tokens)) * idf
    return [t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]]


def _tokenize(text: str) -> list[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text)) if w not in _STOPWORDS]


def token_sets(texts_per_topic: list[list[str]]) -> list[set[str]]:
    """Per-topic token sets, the background for :func:`keywords`."""
    return [set(_tokenize(" ".join(texts))) for texts in texts_per_topic]


def summarize_metrics(
    topics: list[Any],
    *,
    total_chunks: int,
    total_entities: int,
    newman_q: float,
    native_modularity: float | None,
) -> dict[str, Any]:
    """Global grades for one build — the numbers a sweep compares."""
    covered = {cid for t in topics for cid in t.chunk_ids}
    clustered = sum(t.size for t in topics)
    coherences = [t.coherence for t in topics if t.size > 1]
    return {
        "topics": len(topics),
        "newman_modularity": round(newman_q, 4),
        # Reported for continuity, NOT an objective — see the module docstring.
        "arcadedb_modularity_untrustworthy": (
            round(native_modularity, 4) if native_modularity is not None else None
        ),
        "chunk_coverage": round(len(covered) / total_chunks, 4) if total_chunks else 0.0,
        "entity_coverage": round(clustered / total_entities, 4) if total_entities else 0.0,
        "mean_coherence": round(float(np.mean(coherences)), 4) if coherences else 0.0,
        "mean_size": round(clustered / len(topics), 2) if topics else 0.0,
        "largest_topic": max((t.size for t in topics), default=0),
        "singleton_topics": sum(1 for t in topics if t.size <= 1),
    }


__all__ = [
    "coherence",
    "keywords",
    "newman_modularity",
    "summarize_metrics",
    "token_sets",
]
