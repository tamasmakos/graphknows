"""Frame-aware re-ranking of the fused pool.

The measured problem this exists for: on conv-30 the gold chunk is *found* and
then *dropped*. Evidence recall at the probe is 0.948; by the time the pool is
cut to the passages the generator sees it is 0.765, and 13 of 18 failures are
chunks that sat in the pool with the answer in them. Recall is not the
bottleneck — order is.

The FRAME_INSTANCE layer can fix order because it says something the lexical and
vector channels cannot:

- **specificity** — with one instance per trigger, frame frequency is a real
  statistic. ``Self_motion`` covers half of conv-30's chunks and means nothing;
  a frame occurring twice is a fingerprint. That is IDF, and it is graded where
  the saturation guard is binary.
- **role binding** — not "this chunk mentions Jon and evokes Giving" but "Jon
  FILLS A ROLE in a Giving instance here". The reified layer is the only place
  that conjunction can be asked, and it is what separates the gold chunk from
  forty that merely name Jon.

Multiplicative and bounded: RRF scores live on their own scale, so a factor
keeps this scale-free, and the cap means frames reorder *within* the fused
ranking rather than overriding it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any

# A chunk with a perfect frame signal ranks as if its fused score were this much
# higher. At 0.5 a strong match climbs roughly 15-20 places at the depths that
# matter (RRF decays slowly: rank 25 scores 0.0118, rank 40 scores 0.0100), which
# is the size of the observed probe-to-context gap. Higher would let the frame
# layer overrule agreement between four other channels.
ALPHA = float(os.environ.get("GRAPHKNOWS_FRAME_RERANK_ALPHA", "0.5"))
# A role-bound match is worth this many plain frame matches. Binding is the rarer
# and more precise signal, so it dominates without erasing the frame term.
ROLE_BINDING_WEIGHT = 2.0


def enabled() -> bool:
    """Whether frame re-ranking is applied (default on).

    Read at call time so an eval A/B can flip ``GRAPHKNOWS_FRAME_RERANK`` per
    subprocess.
    """
    return os.environ.get("GRAPHKNOWS_FRAME_RERANK", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def idf(df: int, total: int) -> float:
    """Inverse chunk-frequency of a frame within the session, floored at 0.

    A frame in every chunk carries no information and must contribute nothing;
    without the floor, ``log`` of a ratio below 1 goes negative and a saturated
    frame would actively *demote* the chunk it appears in.
    """
    if total <= 0 or df <= 0:
        return 0.0
    return max(0.0, math.log(total / df))


def score_chunks(
    rows: Iterable[dict[str, Any]],
    df: dict[str, int],
    total_chunks: int,
) -> dict[str, float]:
    """Raw frame score per chunk from ``{chunk_id, frame, anchor_hits}`` rows.

    Each matched instance contributes its frame's IDF, multiplied when the query's
    own entities fill roles in it.
    """
    out: dict[str, float] = {}
    for row in rows:
        cid = str(row.get("chunk_id") or "")
        frame = str(row.get("frame") or "")
        if not cid or not frame:
            continue
        weight = idf(df.get(frame, 0), total_chunks)
        if int(row.get("anchor_hits") or 0) > 0:
            weight *= ROLE_BINDING_WEIGHT
        out[cid] = out.get(cid, 0.0) + weight
    return out


def apply(
    combined: dict[str, tuple[float, dict[str, Any]]],
    frame_scores: dict[str, float],
    alpha: float = ALPHA,
) -> dict[str, tuple[float, dict[str, Any]]]:
    """Scale fused scores by their normalised frame signal.

    Normalised against the pool's own best, so the boost is relative to what this
    query found rather than to an absolute scale that varies per conversation.
    Chunks with no frame signal keep their score exactly — this only ever
    promotes, never demotes, so a frame layer that says nothing costs nothing.
    """
    if not frame_scores or not combined:
        return combined
    best = max(frame_scores.values())
    if best <= 0:
        return combined
    out: dict[str, tuple[float, dict[str, Any]]] = {}
    for cid, (score, info) in combined.items():
        boost = 1.0 + alpha * (frame_scores.get(cid, 0.0) / best)
        if boost > 1.0:
            info = {**info, "sources": set(info.get("sources") or ()) | {"frame_rank"}}
        out[cid] = (score * boost, info)
    return out
