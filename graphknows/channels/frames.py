"""FrameChannel — FrameNet semantic-frame retrieval signal.

Chunks are frame-tagged at ingest by cosine-matching their embedding to the
FrameNet frame-definition index (see :mod:`graphknows.symbolic.framenet.index`); each evoked FRAME node
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
from typing import Any

from graphknows.channels._shared import (
    drop_saturated,
    rank_chunks,
    session_chunk_count,
    tag_coverage,
)
from graphknows.channels.base import Channel, ChannelContext, CollectResult

log = logging.getLogger(__name__)

# A frame evoked in more than this fraction of a session's chunks is saturated:
# like the conversation participants (see the entity-saturation drop), a generic
# frame such as ``Activity_ongoing`` matches nearly every chunk, so returning its
# chunks floods recall with the whole conversation and carries no discriminative
# signal. Such query-frames are dropped from the join.
_FRAME_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_FRAME_SATURATION_RATIO", "0.5"))


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

        return rank_chunks(rows, top_k, "frame")

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose frame is evoked in more than half the session's chunks.

        Session-scoped only; without a session there is no corpus to measure
        saturation against, so the rows pass through unchanged.

        Coverage comes from the GRAPH, never from ``rows``. Counting the sample
        made this guard unable to fire at all: on conv-30, ``Activity_done_state``
        covers 19 of session s1's 28 chunks (68%) but showed up on only 8 of any
        query's 25 rows (29%), so it cleared the 50% test on every query and was
        never dropped.

        Scoped by ``c.session_id``: FRAME is canonical — one vertex per frame for
        the whole namespace — so it carries no session of its own. The session is
        a property of the chunk that evoked it.
        """
        if not ctx.session_id:
            return rows
        total = await session_chunk_count(rt, ctx.session_id, ctx.state)
        cover = await tag_coverage(
            rt,
            ctx.session_id,
            ctx.state,
            pattern="(f:FRAME)-[:EVOKED_BY]->(c:CHUNK)",
            key="f.frame_type",
        )
        return drop_saturated(
            rows,
            field="frame_type",
            coverage=cover,
            total=total,
            ratio=_FRAME_SATURATION_RATIO,
            label="frame",
        )
