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

    # Only a literal True enables frames (defensive: a mock/partial settings must
    # not silently pull in the frame channel and its FrameNet load).
    if getattr(settings, "enable_frames", False) is True:
        from graphknows.memory.channels.frame import FrameChannel
        channels.append(FrameChannel())

    mode_obj = getattr(settings, "mode", None)
    mode = getattr(mode_obj, "value", mode_obj)
    if not isinstance(mode, str):
        return channels  # unresolved mode (e.g. a test mock) — do not filter
    return [c for c in channels if mode in c.profiles]
