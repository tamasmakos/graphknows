"""channels — the plugin axis for optional memory signals.

Each channel implements the symmetric :class:`~graphknows.channels.base.Channel`
contract (write at ingest/flush, read at query).
:func:`~graphknows.channels.registry.core_channels` is the fixed core set; it
runs on both the ingestion and retrieval paths.
"""
