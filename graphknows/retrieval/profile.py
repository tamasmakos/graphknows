"""The retrieval knobs a pack may set — and the core defaults when none does.

A profile is data, not behaviour: it says how wide to pool, how far to walk,
whether to gate on dates and how hard to boost frames. The RRF spine in
``graphknows.ranking.rrf`` is fixed and deliberately absent here — rank fusion
is core (FR-001), not a per-pack choice.

Pack supply arrives with the dialogue pack (FR-022); until then every recall
runs on :data:`CORE_PROFILE`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalProfile(BaseModel, frozen=True):
    """How one pack wants the read path shaped.

    Attributes:
        pool_factor: Retrieve ``pool_factor * top_k`` candidates before the cut.
        min_pool: Floor on that pool, so a small ``top_k`` still fuses widely.
        neighbor_radius: Segments either side of a hit stitched back in. 0
            returns the hit alone.
        date_gate: Restrict a date-asking query to time-anchored candidates.
        frame_alpha: Weight of the frame-aware re-rank. 0.0 disables it.
        channels: Extra channel names this profile turns on, over the fixed
            core collectors. Empty means core only.
    """

    pool_factor: int = Field(default=4, ge=1)
    min_pool: int = Field(default=100, ge=0)
    neighbor_radius: int = Field(default=0, ge=0)
    date_gate: bool = True
    frame_alpha: float = Field(default=0.5, ge=0.0)
    channels: frozenset[str] = frozenset()


CORE_PROFILE = RetrievalProfile()
"""The profile a recall runs on when no pack supplies one."""
