"""Build the active retrieval/ingest channel set from settings.

Explicit registry, no auto-discovery: what runs is what's registered here,
filtered by the memory mode each channel declares in ``profiles``, by opt-in
flags, and optionally by an explicit ``enabled`` name list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from graphknows.memory.channels.base import Channel
from graphknows.memory.channels.collectors import DEFAULT_COLLECTOR_CHANNELS

if TYPE_CHECKING:
    from graphknows.config import GraphKnowsSettings

# name -> factory. Callers select channels by name via build_channels(enabled=...).
CHANNEL_REGISTRY: dict[str, type[Channel]] = {cls.name: cls for cls in DEFAULT_COLLECTOR_CHANNELS}


def build_channels(
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
        from graphknows.memory.channels.frame import FrameChannel

        registry[FrameChannel.name] = FrameChannel

    if enabled is None:
        # Default order matches the original hardcoded gather() (registry preserves
        # insertion order), with frames appended when enabled.
        selected = list(registry.values())
    else:
        selected = [registry[name] for name in enabled]

    channels = [cls() for cls in selected]
    mode = settings.mode.value
    return [c for c in channels if mode in c.profiles]
