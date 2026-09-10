"""OntologyChannel — the concept-index retrieval signal.

Segments are concept-labelled at ingest: every mention's pack label resolves a
CONCEPT and the segment is written with an ``EVOKES`` edge to it. Each CONCEPT
carries the embedding of its own definition, so this channel matches the query
with a namespace-local vector search over the concept index and follows
``EVOKES`` back to the segments that evoked the matches — the top-down half of
the traversal, read through one declared store read (``segments_evoking``).

Concepts give abstract questions a route by the KIND of thing a segment is
about — "where did they go" reaches segments through their Place-typed
mentions without the word "place" appearing anywhere. Fully embedding-based,
so valid in both profiles.

Additive to the RRF fusion rather than a re-rank: re-ranking the fused list on a
graph signal has regressed retrieval here before (see the community-detection
experiment), whereas contributing candidates cannot displace a strong hit.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.channels._shared import (
    drop_saturated,
    rank_segments,
    segment_count,
    tag_coverage,
)
from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.settings import get_settings

log = logging.getLogger(__name__)

# Concepts considered per query.
_CONCEPT_TOP_K: int = 5


class OntologyChannel(Channel):
    """Concept-index retrieval channel (both profiles; inert without concepts)."""

    name = "ontology"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Vector-search the namespace's concepts and rank the segments they label."""
        rows = await rt.store.segments_evoking(ctx.emb, _CONCEPT_TOP_K, ctx.source_ids)
        if not rows:
            return {}

        rows = await self._drop_saturated(rows, ctx, rt)
        if not rows:
            return {}
        return rank_segments(rows, top_k, "ontology")

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose concept labels more than half the scope's segments.

        Coverage comes from the GRAPH: counted from ``rows``, the numerator is
        bounded by the page size while the denominator is the whole scope, so
        the ratio could not reach the threshold and this guard never dropped
        anything.

        Source-scoped only; without a source there is no corpus to measure
        saturation against, so the rows pass through unchanged.
        """
        if not ctx.source_ids:
            return rows
        total = await segment_count(rt, ctx.source_ids)
        cover = await tag_coverage(
            rt,
            ctx.source_ids,
            pattern="(g:SEGMENT)-[:EVOKES]->(c:CONCEPT)",
            key="c.uri",
        )
        return drop_saturated(
            rows,
            field="uri",
            coverage=cover,
            total=total,
            ratio=get_settings().class_saturation_ratio,
            label="ontology",
        )
