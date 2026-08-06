"""FrameChannel — FrameNet semantic-frame retrieval signal.

Chunks are frame-tagged at ingest by cosine-matching their embedding to the
FrameNet frame-definition index (see ``_frame_index``); each evoked FRAME node
carries the frame's definition embedding, so this channel matches the query with a
namespace-local graph vector search over those embeddings:
collect() ANN-searches the query embedding against the FRAME.embedding of frames
evoked in this namespace (GraphStore.query_frames_by_embedding), which joins
FRAME-[:EVOKED_BY]->CHUNK, ranks the chunks by their frame's match score, and
emits RRF candidates.

Frames give abstract/general questions ("what did they start?") a route to
concrete chunks by the frame they evoke, complementing the lexical/vector
channels. Fully embedding-based, so the channel is valid in both profiles.

Schema: the FRAME vertex/EVOKED_BY edge live in the GraphStore core DDL.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

# A frame evoked in more than this fraction of a session's chunks is saturated:
# like the conversation participants (see the entity-saturation drop), a generic
# frame such as ``Activity_ongoing`` matches nearly every chunk, so returning its
# chunks floods recall with the whole conversation and carries no discriminative
# signal. Such query-frames are dropped from the join.
_FRAME_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_FRAME_SATURATION_RATIO", "0.5"))


async def _frame_coverage(rt: Any, session_id: str, state: str | None) -> dict[str, int]:
    """How many of the session's chunks each frame_type is evoked in.

    Measured over the GRAPH, not over the rows a query happened to return.
    Counting coverage from the returned rows bounds the numerator by the sample
    size while the denominator stays the whole session, so the ratio could never
    reach the threshold and the guard never fired.

    Scoped by ``c.session_id``: FRAME is canonical — one vertex per frame for
    the whole namespace — so it carries no session of its own. The session is a
    property of the chunk that evoked it.
    """
    q = "MATCH (f:FRAME)-[:EVOKED_BY]->(c:CHUNK) WHERE c.session_id = $sid"
    if state:
        q += " AND c.state = $st"
    q += " RETURN f.frame_type AS ft, count(DISTINCT c.chunk_id) AS n"
    rows = (
        await rt.store.query(q, sid=session_id, st=state)
        if state
        else await rt.store.query(q, sid=session_id)
    )
    return {r["ft"]: int(r.get("n") or 0) for r in rows if r.get("ft")}


async def _session_chunk_count(rt: Any, session_id: str, state: str | None) -> int:
    """Total CHUNK count for a session (optionally filtered by lifecycle state)."""
    if state:
        rows = await rt.store.query(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid AND c.state = $st RETURN count(c) AS c",
            sid=session_id,
            st=state,
        )
    else:
        rows = await rt.store.query(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid RETURN count(c) AS c",
            sid=session_id,
        )
    return int(rows[0]["c"]) if rows else 0


class FrameChannel(Channel):
    """Deterministic FrameNet semantic-frame retrieval channel (both profiles)."""

    name = "frame"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Vector-search the namespace-local FRAME summaries and rank the chunks they evoke."""
        # Namespace-local graph vector search over the FRAME.embedding of frames
        # actually evoked in this namespace (each carries its definition embedding),
        # replacing the global numpy lookup. Chunks are scored by their frame's
        # query-match cosine, so chunks evoking the strongest frames rank first.
        rows = await rt.store.query_frames_by_embedding(
            ctx.emb, top_k, ctx.session_id or None, ctx.state
        )
        if not rows:
            return {}

        rows = await self._drop_saturated(rows, ctx, rt)
        if not rows:
            return {}

        chunk_score: dict[str, float] = defaultdict(float)
        for r in rows:
            chunk_score[r["chunk_id"]] += r.get("score", 0.0)

        ranked = sorted(chunk_score, key=lambda c: -chunk_score[c])[:top_k]
        result: CollectResult = {}
        for rank, cid in enumerate(ranked):
            result[cid] = (
                rrf_score(rank),
                {"text": "", "entities": [], "sources": {"frame"}, "doc_id": "", "metadata": {}},
            )
        return result

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose frame is evoked in >50% of the session's chunks.

        Session-scoped only; without a session there is no corpus to measure
        saturation against, so the rows pass through unchanged.

        Coverage comes from the graph, never from ``rows`` — see
        :func:`_frame_coverage`. Counting the sample made this guard unable to
        fire at all: on conv-30, ``Activity_done_state`` covers 19 of session
        s1's 28 chunks (68%) but showed up on only 8 of any query's 25 rows
        (29%), so it cleared the 50% test on every query and was never dropped.
        """
        if not ctx.session_id:
            return rows
        total = await _session_chunk_count(rt, ctx.session_id, ctx.state)
        if not total:
            return rows
        cover = await _frame_coverage(rt, ctx.session_id, ctx.state)
        present = {r["frame_type"] for r in rows}
        saturated = {ft for ft in present if cover.get(ft, 0) / total > _FRAME_SATURATION_RATIO}
        if not saturated:
            return rows
        log.debug("frame channel dropping saturated frames: %s", sorted(saturated))
        return [r for r in rows if r["frame_type"] not in saturated]
