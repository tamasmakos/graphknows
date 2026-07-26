"""TopicChannel — community-topic retrieval signal.

Topics are built by the topic stage (see ``graphknows.topics``) and carry a
centroid embedding, TF-IDF keywords and a coherence score. This channel uses all
three:

- ANN-match the query against the topic centroids,
- traverse ``CHUNK-[:HAS_TOPIC]->TOPIC`` so a matched topic routes to the real
  chunks it was built from, discounted by the topic's coherence,
- inject the top topic's summary as one synthetic passage, which is the only
  abstractive context the deterministic pipeline produces.

Where the vector channel matches a query to individual chunks, topics match it
to a *theme* and pull in members a lexical or single-chunk vector match would
miss. Additive to RRF rather than a re-rank — re-ranking a fused list on a graph
signal has regressed retrieval in this codebase before.

Data-gated: inert unless the topic layer was built (``GRAPHKNOWS_TOPICS``).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

# A topic covering more than this fraction of a session's chunks is not a theme,
# it is the conversation. Returning its members floods recall with everything —
# the same failure mode as saturated speaker entities, generic frames and the
# ubiquitous Person ontology class.
_TOPIC_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_TOPIC_SATURATION_RATIO", "0.5"))

# Topic centroids considered per query.
_TOPIC_TOP_K: int = 3

# The summary passage is abstractive context rather than retrieved evidence, so
# it is boosted but kept to one.
_SUMMARY_BOOST: float = 1.5


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


class TopicChannel(Channel):
    """Topic-centroid retrieval channel (both profiles; inert without topics)."""

    name = "topic"
    profiles = frozenset({"llm_free", "llm_assisted"})

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Match topic centroids, then rank the member chunks they route to."""
        rows = await rt.store.search_topics_with_chunks(
            ctx.emb, _TOPIC_TOP_K, ctx.session_id or None, ctx.state
        )
        result: CollectResult = {}
        if rows:
            rows = await self._drop_saturated(rows, ctx, rt)
        if rows:
            chunk_score: dict[str, float] = defaultdict(float)
            for r in rows:
                chunk_score[r["chunk_id"]] += float(r.get("score") or 0.0)
            ranked = sorted(chunk_score, key=lambda c: -chunk_score[c])[:top_k]
            for rank, cid in enumerate(ranked):
                result[cid] = (
                    rrf_score(rank),
                    {
                        "text": "",
                        "entities": [],
                        "sources": {"topic"},
                        "doc_id": "",
                        "metadata": {},
                    },
                )
        await self._add_summary(result, ctx, rt)
        return result

    @staticmethod
    async def _add_summary(result: CollectResult, ctx: ChannelContext, rt: Any) -> None:
        """Inject the single best-matching topic summary as a synthetic passage."""
        topics = await rt.store.search_topics(ctx.emb, top_k=1)
        for rank, topic in enumerate(topics):
            summary = (topic.get("summary") or "").strip()
            tid = topic.get("topic_id") or ""
            if not summary:
                continue
            keywords = ", ".join(topic.get("keywords") or [])
            text = f"{summary} [topic keywords: {keywords}]" if keywords else summary
            result[f"topic:{tid}"] = (
                rrf_score(rank, boost=_SUMMARY_BOOST),
                {
                    "text": text,
                    "entities": [],
                    "sources": {"topic"},
                    "doc_id": "",
                    "metadata": {"is_topic_summary": True},
                },
            )

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop topics covering >50% of the session's chunks.

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
            cover[r["topic_id"]].add(r["chunk_id"])
        saturated = {
            tid for tid, chunks in cover.items() if len(chunks) / total > _TOPIC_SATURATION_RATIO
        }
        if not saturated:
            return rows
        log.debug("topic channel dropping saturated topics: %s", sorted(saturated))
        return [r for r in rows if r["topic_id"] not in saturated]
