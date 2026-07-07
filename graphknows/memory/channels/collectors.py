"""Adapter channels for the seven existing DETRetriever collectors.

These are transitional: each ``collect`` calls the retriever's original,
battle-tested collector method unchanged, so the channel seam is introduced with
byte-identical retrieval behavior. The collector bodies can migrate into these
classes later; the seam is what matters now.
"""
from __future__ import annotations

from typing import Any

from graphknows.memory.channels.base import Channel, ChannelContext, CollectResult


class StmEntityChannel(Channel):
    """STM entity-graph boost — chunks reached from query-token entity nodes."""

    name = "stm_entity"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's STM entity collector."""
        return await rt.stm_entity_boost(ctx.words_graph, ctx.session_id, top_k)


class LtmEntityChannel(Channel):
    """LTM entity-graph traversal from the query's discriminative entities."""

    name = "ltm_entity"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's LTM entity-traversal collector."""
        return await rt.ltm_entity_traversal(ctx.words_graph, ctx.session_id, top_k)


class VectorChannel(Channel):
    """Dense vector ANN over chunk embeddings (STM + LTM)."""

    name = "vector"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's semantic ANN collector."""
        return await rt.semantic_ann(ctx.emb, ctx.session_id, ctx.scope, top_k, emb2=ctx.emb2)


class StmBm25Channel(Channel):
    """Lexical full-text (BM25) search over STM chunks."""

    name = "stm_bm25"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's STM full-text collector."""
        return await rt.stm_bm25(ctx.words, ctx.session_id, top_k)


class LtmBm25Channel(Channel):
    """Lexical full-text (BM25) search over LTM chunks."""

    name = "ltm_bm25"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's LTM full-text collector."""
        return await rt.ltm_bm25(ctx.words, ctx.session_id, ctx.scope, top_k)


class TemporalChannel(Channel):
    """Temporal traversal — chunks anchored to dates resolved from the query."""

    name = "temporal"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's temporal-traversal collector."""
        return await rt.ltm_temporal_traversal(ctx.query, ctx.session_id, ctx.scope, top_k)


class TopicChannel(Channel):
    """Topic-summary match (llm_assisted topics; empty in llm_free)."""

    name = "topic"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Delegate to the retriever's topic-match collector."""
        return await rt.ltm_topic_match(ctx.emb, ctx.session_id, ctx.scope, top_k)


# The default retrieval channel set, in fusion order (matches the original
# hardcoded gather()).  The builder appends optional channels (e.g. frames).
DEFAULT_COLLECTOR_CHANNELS: list[type[Channel]] = [
    StmEntityChannel, LtmEntityChannel, VectorChannel,
    StmBm25Channel, LtmBm25Channel, TemporalChannel, TopicChannel,
]
