"""FrameChannel — deterministic FrameNet semantic-frame retrieval signal.

Owned end-to-end:
- promote(): at flush, extract frames from the session's LTM chunks and write a
  FRAME layer (one node per frame evoked in a chunk, with FE fillers + speaker).
- collect(): parse the query into frames + anchors, match FRAME nodes by frame
  type (ontology-expanded), rank chunks by an instance-join score (exact-frame
  weight x anchor-in-role bonus), and emit RRF candidates.

Abstract-question gate: when the query evokes no concrete frame (pure inference
questions like "would X be religious"), collect() returns nothing — frames only
help where there's structure to match, and staying silent avoids displacing the
vector channel on those questions.

Schema: the FRAME vertex/indexes live in ltm_store._SCHEMA_DDL (ensured on
connect). Extraction cost is ~spaCy+embed per sentence, run once per session at
flush; fully local/API-free, so the channel is valid in both profiles.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

_EXACT_FRAME_WEIGHT = 2.0
_EXPANDED_FRAME_WEIGHT = 1.0
_ANCHOR_BONUS = 1.6
_SPEAKER_LINE = re.compile(r"^.*?\|\s*(\w+):")


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

        rows = await rt.ltm._qry(
            "MATCH (c:CHUNK {session_id: $sid}) RETURN c.id AS id, c.text AS text",
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

        written = await rt.ltm.write_frames(session_id, frame_rows)
        log.info("FrameChannel.promote session=%s chunks=%d frames=%d",
                 session_id, len(chunks), written)
        if record is not None and hasattr(record, "audit"):
            # non-fatal diagnostic; mirrors how other phases annotate the record
            with_frames = getattr(record.audit, "frames_written", None)
            if with_frames is not None:
                record.audit.frames_written = written

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Match FRAME nodes to the query's frames and rank chunks (instance-join)."""
        parse = ctx.frame_parse()
        q_frames: set[str] = parse["q_frames"]
        if not q_frames:
            return {}  # abstract-question gate: no concrete frame to match
        expanded: set[str] = parse["expanded"]
        anchors: set[str] = parse["anchors"]

        rows = await rt.ltm.query_frame_chunks(ctx.session_id, list(expanded))
        if not rows:
            return {}

        chunk_score: dict[str, float] = defaultdict(float)
        for r in rows:
            base = _EXACT_FRAME_WEIGHT if r["frame_type"] in q_frames else _EXPANDED_FRAME_WEIGHT
            inst_ents = set(r["fillers"]) | ({r["speaker"]} if r["speaker"] else set())
            if anchors & inst_ents:
                base *= _ANCHOR_BONUS
            chunk_score[r["chunk_id"]] += base

        ranked = sorted(chunk_score, key=lambda c: -chunk_score[c])[:top_k]
        result: CollectResult = {}
        for rank, cid in enumerate(ranked):
            result[cid] = (
                rrf_score(rank),
                {"text": "", "entities": [], "sources": {"frame"},
                 "doc_id": "", "metadata": {}},
            )
        return result
