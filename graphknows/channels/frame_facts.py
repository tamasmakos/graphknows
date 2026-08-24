"""FrameFactChannel — route the fact layer back to the chunks it was read from.

Distinct from :class:`~graphknows.channels.frames.FrameChannel`, which matches
the query embedding against the FRAME chunk-TAG layer. This one reads the
FRAME_INSTANCE layer: the reified ``(instance)-[:PLAYS_ROLE]->(entity)`` facts
that until now only ever reached the generator's fact-sheet, never retrieval.

Why it can work where ``ppr`` did not: ``REL`` (the relation layer) has no edge
to a chunk, so a graph walk over it can only return entities and the retriever
must guess which chunk stated them — measured 3.3% gold-bearing on conv-30. A
FRAME_INSTANCE carries ``FRAME_EVOKED_IN`` to the chunk it was extracted from,
so a role match names its own evidence.

DEFAULT OFF (``enable_frame_facts``). Adding a leg to the fusion is the move
that has cost this project before — the ontology channel's sixth leg tracked
accuracy 0.766 -> 0.722 — so this ships as an A/B, not as a default.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.channels._shared import session_chunk_count
from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.channels.frames import _FRAME_SATURATION_RATIO
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)


async def _instance_coverage(
    rt: Any, frames: list[str], session_id: str, state: str | None
) -> dict[str, int]:
    """How many of the session's chunks each of ``frames`` has an INSTANCE in.

    The FRAME_INSTANCE layer, not the FRAME chunk-tag layer that
    :func:`~graphknows.channels._shared.tag_coverage` measures — SRL writes one
    instance per trigger, so the two layers have very different densities and a
    frame saturated in one need not be in the other.

    Scoped to the query's own frames: coverage for the rest of the vocabulary is
    never read, so this stays a bounded lookup rather than a whole-layer scan.
    """
    from graphknows.storage.arcadedb._sql import quote, quoted_list

    scope = f" AND c.state = {quote(state)}" if state else ""
    rows = await rt.store.query(
        f"MATCH (fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c:CHUNK) "
        f"WHERE fi.frame IN {quoted_list(sorted(set(frames)))} "
        f"AND c.session_id = {quote(session_id)}{scope} "
        f"RETURN fi.frame AS ft, count(DISTINCT c.chunk_id) AS n"
    )
    return {r["ft"]: int(r.get("n") or 0) for r in rows if r.get("ft")}


class FrameFactChannel(Channel):
    """Chunks whose frame instances match the query's frames or entities."""

    name = "frame_fact"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Match query frames + resolved entities against FRAME_INSTANCE, rank their chunks."""
        parse = ctx.frame_parse()
        # The expanded set (1 hop up the FrameNet hierarchy) rather than the bare
        # query frames: the question and the statement rarely evoke the SAME
        # frame — "what did she give him" is Giving, the chunk may be Receiving
        # or Commerce_pay under a shared parent.
        frames = sorted(parse.get("expanded") or parse.get("q_frames") or ())
        norms = await rt.store.resolve_query_entities(ctx.words_graph)
        if not frames and not norms:
            return {}

        frames = await self._drop_saturated(frames, ctx, rt)
        if not frames and not norms:
            return {}

        rows = await rt.store.query_chunks_by_frame_facts(
            frames, norms, top_k, ctx.session_id or None, ctx.state
        )
        result: CollectResult = {}
        for rank, row in enumerate(rows):
            cid = row.get("chunk_id")
            if not cid:
                continue
            result[str(cid)] = (
                rrf_score(rank),
                {
                    "text": "",
                    "entities": [],
                    "sources": {self.name},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    @staticmethod
    async def _drop_saturated(frames: list[str], ctx: ChannelContext, rt: Any) -> list[str]:
        """Drop query frames with an instance in >50% of the session's chunks.

        The sibling channels both learned this the hard way; this one shipped
        without it, and SRL is what made the omission bite. With one instance
        per trigger instead of at most one per chunk, ``Self_motion`` alone
        reaches 54 of conv-30's 110 chunks — so the join returned essentially
        the whole conversation and the RRF pool ranked gold below noise.

        Filtered BEFORE the join rather than after, so a saturated frame costs
        nothing to reject. Session-scoped only: with no session there is no
        corpus to measure a ratio against, so the frames pass through.
        """
        if not frames or not ctx.session_id:
            return frames
        total = await session_chunk_count(rt, ctx.session_id, ctx.state)
        if not total:
            return frames
        cover = await _instance_coverage(rt, frames, ctx.session_id, ctx.state)
        kept = [f for f in frames if cover.get(f, 0) / total <= _FRAME_SATURATION_RATIO]
        if len(kept) != len(frames):
            log.debug("frame_fact dropping saturated frames: %s", sorted(set(frames) - set(kept)))
        return kept
