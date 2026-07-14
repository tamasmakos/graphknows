"""FrameChannel — FrameNet semantic-frame retrieval signal.

Chunks are frame-tagged at ingest by cosine-matching their embedding to the
FrameNet frame-definition index (see ``_frame_index``); each evoked FRAME node
carries the frame's definition embedding, so this channel matches the query with a
namespace-local graph vector search over those embeddings:
- collect(): ANN-search the query embedding against the FRAME.embedding of frames
  evoked in this namespace (GraphStore.query_frames_by_embedding), which joins
  FRAME-[:EVOKED_BY]->CHUNK, rank the chunks by their frame's match score, and emit
  RRF candidates.
- promote(): dormant LU-based extraction path (frames are now written at ingest,
  so it is no longer invoked on the live flush shim); retained for reference.

Frames give abstract/general questions ("what did they start?") a route to
concrete chunks by the frame they evoke, complementing the lexical/vector
channels. Fully embedding-based, so the channel is valid in both profiles.

Schema: the FRAME vertex/EVOKED_BY edge live in the GraphStore core DDL.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

_SPEAKER_LINE = re.compile(r"^.*?\|\s*(\w+):")

# A frame evoked in more than this fraction of a session's chunks is saturated:
# like the conversation participants (see the entity-saturation drop), a generic
# frame such as ``Activity_ongoing`` matches nearly every chunk, so returning its
# chunks floods recall with the whole conversation and carries no discriminative
# signal. Such query-frames are dropped from the join.
_FRAME_SATURATION_RATIO: float = float(
    os.environ.get("GRAPHKNOWS_FRAME_SATURATION_RATIO", "0.5")
)


async def _session_chunk_count(rt: Any, session_id: str, state: str | None) -> int:
    """Total CHUNK count for a session (optionally filtered by lifecycle state)."""
    if state:
        rows = await rt.store._qry(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid AND c.state = $st "
            "RETURN count(c) AS c",
            sid=session_id, st=state,
        )
    else:
        rows = await rt.store._qry(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid RETURN count(c) AS c",
            sid=session_id,
        )
    return int(rows[0]["c"]) if rows else 0


def _session_speakers(chunk_texts: list[str]) -> list[str]:
    seen: list[str] = []
    for text in chunk_texts:
        for line in text.split("\n"):
            m = _SPEAKER_LINE.match(line.strip())
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
    return seen


class FrameChannel(Channel):
    """Deterministic FrameNet semantic-frame retrieval channel (both profiles)."""

    name = "frame"
    profiles = frozenset({"llm_free", "llm_assisted"})

    async def promote(self, rt: Any, session_id: str, record: Any) -> None:
        """Extract frames from the session's LTM chunks and write the FRAME layer."""
        from graphknows.channels._framenet import extract_frames_from_chunk

        rows = await rt.store._qry(
            "MATCH (c:CHUNK {session_id: $sid}) RETURN c.chunk_id AS id, c.text AS text",
            sid=session_id,
        )
        chunks = [(r["id"], r.get("text") or "") for r in rows if r.get("id")]
        if not chunks:
            return
        speakers = _session_speakers([t for _, t in chunks])

        frame_rows: list[dict[str, Any]] = []
        for cid, text in chunks:
            for inst in extract_frames_from_chunk(text, speakers):
                fes = inst["fes"]
                fillers = [f for _, f in fes]
                if inst["speaker"]:
                    fillers.append(inst["speaker"])
                frame_rows.append({
                    "frame_type": inst["frame"],
                    "chunk_id": cid,
                    "lu": inst["lu"],
                    "speaker": inst["speaker"],
                    "fes": [[r, f] for r, f in fes],
                    "fillers": fillers,
                })

        written = await rt.store.write_frames(session_id, frame_rows)
        log.info("FrameChannel.promote session=%s chunks=%d frames=%d",
                 session_id, len(chunks), written)
        if record is not None and hasattr(record, "audit"):
            # non-fatal diagnostic; mirrors how other phases annotate the record
            with_frames = getattr(record.audit, "frames_written", None)
            if with_frames is not None:
                record.audit.frames_written = written

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
                {"text": "", "entities": [], "sources": {"frame"},
                 "doc_id": "", "metadata": {}},
            )
        return result

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose frame is evoked in >50% of the session's chunks.

        Session-scoped only; without a session there is no corpus to measure
        saturation against, so the rows pass through unchanged.
        """
        if not ctx.session_id:
            return rows
        total = await _session_chunk_count(rt, ctx.session_id, ctx.state)
        if not total:
            return rows
        cover: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            cover[r["frame_type"]].add(r["chunk_id"])
        saturated = {
            ft for ft, cs in cover.items()
            if len(cs) / total > _FRAME_SATURATION_RATIO
        }
        if not saturated:
            return rows
        log.debug("frame channel dropping saturated frames: %s", sorted(saturated))
        return [r for r in rows if r["frame_type"] not in saturated]
