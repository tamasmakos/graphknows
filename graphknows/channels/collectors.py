"""The built-in retrieval collector channels.

Each ``collect`` delegates to the retriever's corresponding collector method, so
the default signals share the retriever's battle-tested query logic. New signals
subclass :class:`~graphknows.channels.base.Channel` in their own module and
register themselves; these stay the deterministic baseline set.

Golden-layer note (Phase 1): the STM/LTM channel split collapsed into one
channel per signal. There is a single database per namespace, so each collector
queries the one :class:`~graphknows.storage.arcadedb.graph_store.GraphStore`
with a lifecycle ``state`` filter (raw / consolidated / both) instead of routing
to a physical STM vs LTM store. This avoids double-counting a chunk across two
channels while keeping fusion order stable.
"""

from __future__ import annotations

from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult


class EntityChannel(Channel):
    """Entity-graph boost — chunks reached from query-token entity nodes."""

    name = "entity"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's entity collector."""
        return await rt.entity_boost(ctx.words_graph, ctx.session_id, ctx.state, top_k)


class VectorChannel(Channel):
    """Dense vector ANN over chunk embeddings."""

    name = "vector"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's semantic ANN collector."""
        return await rt.semantic_ann(ctx.emb, ctx.session_id, ctx.state, top_k, emb2=ctx.emb2)


class Bm25Channel(Channel):
    """Lexical full-text (BM25) search over chunks."""

    name = "bm25"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's BM25 collector."""
        return await rt.bm25(ctx.words, ctx.session_id, ctx.state, top_k)


class TemporalChannel(Channel):
    """Temporal traversal — chunks anchored to dates resolved from the query."""

    name = "temporal"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's temporal-traversal collector.

        ``words_graph`` is passed so a "when did X happen" query — which states
        no date for the date-keyed path to match — can still find dated chunks
        about X.
        """
        return await rt.temporal_traversal(
            ctx.query, ctx.session_id, ctx.state, top_k, ctx.words_graph
        )


# The default retrieval channel set, in fusion order. The registry appends
# optional channels (frames when enabled, ontology when one is configured).
DEFAULT_COLLECTOR_CHANNELS: list[type[Channel]] = [
    EntityChannel,
    VectorChannel,
    Bm25Channel,
    TemporalChannel,
]

__all__ = [
    "DEFAULT_COLLECTOR_CHANNELS",
    "Bm25Channel",
    "EntityChannel",
    "TemporalChannel",
    "VectorChannel",
]
