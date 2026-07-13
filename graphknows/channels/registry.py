"""Channel registry — the single extension point for memory signals.

Register a channel once (``register("metadata", MetadataChannel)``) and it is
available to both the ingestion promotion pass and the retrieval collector fan-out.
``default_channels(settings)`` returns the active set for a configuration,
filtered by each channel's declared ``profiles`` against the memory mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from graphknows.channels.base import Channel
from graphknows.channels.collectors import DEFAULT_COLLECTOR_CHANNELS
from graphknows.channels.graph_ppr import GraphPPRChannel

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings

# name -> channel class. Insertion order defines the default fusion order.
CHANNEL_REGISTRY: dict[str, type[Channel]] = {
    cls.name: cls for cls in DEFAULT_COLLECTOR_CHANNELS
}
# Personalized-PageRank graph channel — seeds on query entities, propagates over
# the relation graph. Fused alongside the deterministic collectors.
CHANNEL_REGISTRY[GraphPPRChannel.name] = GraphPPRChannel


def register(name: str, channel_cls: type[Channel]) -> None:
    """Register (or replace) a channel under ``name``."""
    CHANNEL_REGISTRY[name] = channel_cls


def default_channels(
    settings: GraphKnowsSettings,
    enabled: Sequence[str] | None = None,
) -> list[Channel]:
    """Instantiate the channels active for this configuration.

    Args:
        settings: Runtime settings (drives mode filtering + opt-in flags).
        enabled: Optional explicit channel-name allow-list. When ``None`` the
            default set is used (all registered collectors, plus frames when
            ``settings.enable_frames``). Unknown names raise ``KeyError``.
    """
    registry = dict(CHANNEL_REGISTRY)
    if settings.enable_frames:
        from graphknows.channels.frames import FrameChannel

        registry[FrameChannel.name] = FrameChannel

    selected = list(registry.values()) if enabled is None else [registry[n] for n in enabled]
    mode = settings.mode.value
    return [c for c in (cls() for cls in selected) if mode in c.profiles]


__all__ = ["CHANNEL_REGISTRY", "default_channels", "register"]
