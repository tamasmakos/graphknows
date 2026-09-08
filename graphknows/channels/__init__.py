"""channels — the plugin axis for optional memory signals.

Each channel implements the symmetric :class:`~graphknows.channels.base.Channel`
contract (write at ingest/flush, read at query).
:func:`~graphknows.channels.registry.default_channels` is the one place to turn
a signal on; it then runs on both the ingestion and retrieval paths.
"""
