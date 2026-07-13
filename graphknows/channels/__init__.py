"""channels — the plugin axis for memory signals.

Each channel implements the symmetric :class:`~graphknows.channels.base.Channel`
contract (write at ingest/flush, read at query). The registry is the one place
to add a signal; it then runs on both the ingestion and retrieval paths.
"""

from graphknows.channels.base import Channel, ChannelContext, ChannelStores, CollectResult
from graphknows.channels.registry import CHANNEL_REGISTRY, default_channels, register

__all__ = [
    "CHANNEL_REGISTRY",
    "Channel",
    "ChannelContext",
    "ChannelStores",
    "CollectResult",
    "default_channels",
    "register",
]
