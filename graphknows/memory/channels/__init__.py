"""Modular retrieval/ingest channels.

A *channel* is one retrieval signal owned end-to-end: its storage schema, its
ingest-time write, its flush-time promotion, and its query-time collection. The
DETRetriever spine (RRF fusion, shaping, cross-encoder rerank) is channel-blind —
channels only *provide* ranked candidates; they cannot reweight each other. Add a
new signal by adding one Channel subclass and listing it in the builder — no edits
to the retriever, ingest loop, or flush pipeline.
"""

from graphknows.memory.channels.base import Channel, ChannelContext

__all__ = ["Channel", "ChannelContext"]
