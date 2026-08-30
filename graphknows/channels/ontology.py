"""OntologyChannel — ontology-class retrieval signal.

Chunks are class-tagged at ingest by cosine-matching their embedding to the
ontology's class DEFINITIONS (see ``graphknows.symbolic.ontology.catalog``); each
ONTOLOGY_CLASS node carries that definition's embedding, so this channel matches
the query with a namespace-local graph vector search over them, exactly as
:class:`~graphknows.channels.frames.FrameChannel` does over FRAME nodes.

A matched class reaches chunks by one path:

- ``CHUNK-[:MATCHES_CLASS]->ONTOLOGY_CLASS`` — the chunk itself evoked the
  class. Direct, and the stronger signal.

A second route through ``CHUNK-[:MENTIONS]->ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS``
was measured byte-identical on conv-26 evidence_recall once the pool is
cosine-reranked, and removed — see the comment in ``graph_store.py``'s
``search_ontology_classes``. ``write_groundings`` still writes INSTANCE_OF
edges, but nothing reads them at retrieval today.

Where frames give abstract questions a route by the situation a chunk evokes,
classes give them a route by the KIND of thing a chunk is about — "where did
they go" reaches chunks through their Place-typed entities without the word
"place" appearing anywhere. Fully embedding-based, so valid in both profiles.

Additive to the RRF fusion rather than a re-rank: re-ranking the fused list on a
graph signal has regressed retrieval here before (see the community-detection
experiment), whereas contributing candidates cannot displace a strong hit.

Schema: ONTOLOGY_CLASS + MATCHES_CLASS live in the GraphStore core DDL.
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

# A class matching more than this fraction of a session's chunks carries no
# discriminative signal: in a personal-conversation corpus nearly every chunk is
# "about" a Person, so returning its chunks floods recall with the whole
# conversation. Same failure mode (and same remedy) as saturated speaker
# entities and saturated generic frames.
_CLASS_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_CLASS_SATURATION_RATIO", "0.5"))

# Ontology classes considered per query.
_CLASS_TOP_K: int = 5


class OntologyChannel(Channel):
    """Ontology-class retrieval channel (both profiles; inert without an ontology)."""

    name = "ontology"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Vector-search the namespace's ontology classes and rank the chunks they tag."""
        rows = await rt.store.search_ontology_classes(
            ctx.emb, _CLASS_TOP_K, ctx.session_id or None, ctx.state
        )
        if not rows:
            return {}

        rows = await self._drop_saturated(rows, ctx, rt)
        if not rows:
            return {}
        return rank_chunks(rows, top_k, "ontology")

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose class tags more than half the session's chunks.

        Coverage comes from the GRAPH. It used to be counted from ``rows``, which
        is the defect the frame channel had already found and fixed: the
        numerator is bounded by the page size while the denominator is the whole
        session, so the ratio could not reach the threshold and this guard never
        dropped anything.

        Session-scoped only; without a session there is no corpus to measure
        saturation against, so the rows pass through unchanged.
        """
        if not ctx.session_id:
            return rows
        total = await session_chunk_count(rt, ctx.session_id, ctx.state)
        cover = await tag_coverage(
            rt,
            ctx.session_id,
            ctx.state,
            pattern="(c:CHUNK)-[:MATCHES_CLASS]->(o:ONTOLOGY_CLASS)",
            key="o.uri",
        )
        return drop_saturated(
            rows,
            field="uri",
            coverage=cover,
            total=total,
            ratio=_CLASS_SATURATION_RATIO,
            label="ontology",
        )
