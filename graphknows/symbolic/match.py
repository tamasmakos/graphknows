"""The two-stage matcher's accept gate.

Shared by every channel that ranks a lexically-anchored candidate set by
cosine. The candidate stage differs per channel (FrameNet's lemma+POS lookup,
the ontology's altLabel index) — this is what all of them do once candidates
are ranked: take the top one if it clears a floor and beats the runner-up by
a margin, else abstain. Originally ``framenet._accept_frame``'s body; pulled
out because the ontology channel needed the identical gate and a second copy
of the same six lines is bug reuse, not a second implementation.
"""

from __future__ import annotations

from collections.abc import Mapping


def accept(
    ranked: list[str],
    sim_of: Mapping[str, float],
    *,
    floor: float,
    floor_single: float,
    margin: float,
) -> tuple[str | None, float]:
    """Top candidate passing the floor + margin gates, or ``(None, 0.0)``.

    ``floor_single`` applies when ``ranked`` has exactly one candidate — a lone
    candidate gets no runner-up to be checked against, so it is held to a
    stricter bar instead.
    """
    if not ranked:
        return None, 0.0
    top1 = sim_of[ranked[0]]
    floor_gate = floor_single if len(ranked) == 1 else floor
    if top1 < floor_gate or (len(ranked) > 1 and top1 - sim_of[ranked[1]] < margin):
        return None, 0.0
    return ranked[0], top1
