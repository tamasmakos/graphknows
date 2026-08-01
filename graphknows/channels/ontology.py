"""OntologyChannel — ontology-class retrieval signal.

Chunks are class-tagged at ingest by cosine-matching their embedding to the
ontology's class DEFINITIONS (see ``graphknows.ontology.catalog``); each
ONTOLOGY_CLASS node carries that definition's embedding, so this channel matches
the query with a namespace-local graph vector search over them, exactly as
:class:`~graphknows.channels.frames.FrameChannel` does over FRAME nodes.

A matched class reaches chunks by two paths, both used:

- ``CHUNK-[:MATCHES_CLASS]->ONTOLOGY_CLASS`` — the chunk itself evoked the
  class. Direct, and the stronger signal.
- ``CHUNK-[:MENTIONS]->ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`` — the chunk
  mentions something that *is* an instance of the class. Much denser (on
  conv-26: 767 INSTANCE_OF edges against 189 MATCHES_CLASS), and discounted as
  indirect evidence since the entity may be incidental to the chunk.

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
from collections import defaultdict
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

# A class matching more than this fraction of a session's chunks carries no
# discriminative signal: in a personal-conversation corpus nearly every chunk is
# "about" a Person, so returning its chunks floods recall with the whole
# conversation. Same failure mode (and same remedy) as saturated speaker
# entities and saturated generic frames.
_CLASS_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_CLASS_SATURATION_RATIO", "0.5"))

# Ontology classes considered per query.
_CLASS_TOP_K: int = 5


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

        chunk_score: dict[str, float] = defaultdict(float)
        for r in rows:
            chunk_score[r["chunk_id"]] += float(r.get("score") or 0.0)

        ranked = sorted(chunk_score, key=lambda c: -chunk_score[c])[:top_k]
        result: CollectResult = {}
        for rank, cid in enumerate(ranked):
            result[cid] = (
                rrf_score(rank),
                {"text": "", "entities": [], "sources": {"ontology"}, "doc_id": "", "metadata": {}},
            )
        return result

    @staticmethod
    async def _drop_saturated(
        rows: list[dict[str, Any]], ctx: ChannelContext, rt: Any
    ) -> list[dict[str, Any]]:
        """Drop rows whose class tags >50% of the session's chunks.

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
            cover[r["uri"]].add(r["chunk_id"])
        saturated = {
            uri for uri, chunks in cover.items() if len(chunks) / total > _CLASS_SATURATION_RATIO
        }
        if not saturated:
            return rows
        log.debug("ontology channel dropping saturated classes: %s", sorted(saturated))
        return [r for r in rows if r["uri"] not in saturated]
