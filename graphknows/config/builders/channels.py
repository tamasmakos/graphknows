"""Build the active retrieval/ingest channel set from settings.

Explicit list, no auto-discovery: what runs is what's written here, filtered by
the memory mode each channel declares in ``profiles`` and by opt-in flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphknows.memory.channels.base import Channel
from graphknows.memory.channels.collectors import DEFAULT_COLLECTOR_CHANNELS

if TYPE_CHECKING:
    from graphknows.config import GraphKnowsSettings


def build_channels(settings: GraphKnowsSettings) -> list[Channel]:
    """Instantiate the channels active for this configuration."""
    channels: list[Channel] = [cls() for cls in DEFAULT_COLLECTOR_CHANNELS]

    if settings.enable_frames:
        from graphknows.memory.channels.frame import FrameChannel

        channels.append(FrameChannel())

    mode = settings.mode.value
    return [c for c in channels if mode in c.profiles]
