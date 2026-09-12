"""Bounded candidate generation: who an entity is actually compared against.

Blocking (``blocking.py``) and the vector index over entity definition
embeddings each propose neighbours; this module is where those proposals are
capped (FR-020). A name that blocks into a very large set stays comparable in
bounded time, and the entities dropped by the cap are counted rather than
quietly lost — ``CandidatePool.truncated`` is what ingest reports as
``Counters.candidates_truncated``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

MAX_CANDIDATES = 50
"""Comparisons one entity is worth: past this the marginal candidate never wins."""


@dataclass(frozen=True)
class CandidatePool:
    """The entities one entity is compared against, and what the cap cut.

    Attributes:
        ids: Candidate entity ids, in the order they were proposed.
        truncated: Distinct candidates dropped by the cap.
    """

    ids: tuple[str, ...]
    truncated: int


def bounded_candidates(entity_id: str, neighbours: Iterable[str]) -> CandidatePool:
    """Cap *neighbours* to :data:`MAX_CANDIDATES` comparisons for *entity_id*.

    *neighbours* is the proposal stream in priority order — blocked entities
    first, then vector-index neighbours — so the cap drops the weakest
    proposals. The entity itself and repeats (an entity both blocks and
    neighbours) never spend a comparison.
    """
    kept: list[str] = []
    seen = {entity_id}
    truncated = 0
    for candidate in neighbours:
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(kept) < MAX_CANDIDATES:
            kept.append(candidate)
        else:
            truncated += 1
    return CandidatePool(ids=tuple(kept), truncated=truncated)
