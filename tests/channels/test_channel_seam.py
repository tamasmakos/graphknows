"""The custom-schema seam: one Channel object carries both hooks.

A caller extends the graph by handing ``Memory(channels=[...])`` an object
that writes its own structure — ``populate`` — and reads it back during recall
as one more leg of the fusion — ``collect``. The dead per-chunk ``ingest`` hook
is gone: it was declared, documented, and never called by anything.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from graphknows.channels.base import Channel


class _RecordingChannel(Channel):
    name = "recording"

    def __init__(self) -> None:
        self.populated: list[tuple[Any, str]] = []

    async def populate(self, store: Any, session_id: str) -> None:
        self.populated.append((store, session_id))


def test_channel_has_no_dead_ingest_hook() -> None:
    """``ingest`` was a documented no-op nothing ever called — the repo's
    named defect class. The write hook is ``populate``."""
    assert not hasattr(Channel, "ingest")
    assert hasattr(Channel, "populate")


@pytest.mark.asyncio
async def test_default_populate_is_a_noop() -> None:
    await Channel().populate(MagicMock(), "s1")


def test_build_retriever_appends_caller_channels() -> None:
    from graphknows.retrieval import build_retriever
    from graphknows.settings import GraphKnowsSettings

    ch = _RecordingChannel()
    retriever = build_retriever(GraphKnowsSettings(), store=MagicMock(), extra_channels=[ch])

    assert ch in retriever._channels
