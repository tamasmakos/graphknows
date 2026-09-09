"""The custom-schema seam: one Channel object carries both hooks.

A caller extends the graph by handing ``Memory(channels=[...])`` an object
that (a) writes its own structure once per flush, after the turn buffer has
drained into CHUNKs — ``populate`` — and (b) reads it back during recall as
one more leg of the fusion — ``collect``, which already existed. The dead
per-chunk ``ingest`` hook is gone: it was declared, documented, and never
called by anything.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.channels.base import Channel
from graphknows.memory import Memory


class _RecordingChannel(Channel):
    name = "recording"

    def __init__(self) -> None:
        self.populated: list[tuple[Any, str]] = []

    async def populate(self, store: Any, session_id: str) -> None:
        self.populated.append((store, session_id))


def test_channel_has_no_dead_ingest_hook() -> None:
    """``ingest`` was a documented no-op nothing ever called — the repo's
    named defect class. The write hook is ``populate``, once per flush."""
    assert not hasattr(Channel, "ingest")
    assert hasattr(Channel, "populate")


@pytest.mark.asyncio
async def test_default_populate_is_a_noop() -> None:
    await Channel().populate(MagicMock(), "s1")


def _memory_with_fake_store(channels: list[Channel]) -> tuple[Memory, MagicMock]:
    """A Memory whose store layer is a mock — flush's steps soft-fail around it."""
    mem = Memory(namespace="t", channels=channels)
    store = MagicMock()
    store.list_turns = AsyncMock(return_value=[])
    store.mark_turns_consolidated = AsyncMock(return_value=0)
    store._ensure_analytical_view = AsyncMock()
    store.compute_and_persist_pagerank = AsyncMock(return_value=0)
    store.compute_graph_embeddings = AsyncMock(return_value=0)
    store.compute_communities = AsyncMock(return_value=0)
    store.consolidate_session = AsyncMock(return_value=0)
    store.query = AsyncMock(return_value=[])
    store.command = AsyncMock(return_value=[])
    mem._store = store
    mem._retriever = MagicMock()  # _get_retriever short-circuits
    return mem, store


@pytest.mark.asyncio
async def test_flush_calls_populate_with_the_store_and_session() -> None:
    """The ordering contract: populate runs inside flush, when CHUNKs exist.

    Called any earlier there is nothing to attach custom edges to; a channel
    that wrote at add() time would MERGE against chunks not yet minted.
    """
    ch = _RecordingChannel()
    mem, store = _memory_with_fake_store([ch])

    await mem.flush_memory("s1")

    assert ch.populated == [(store, "s1")]


@pytest.mark.asyncio
async def test_a_failing_populate_is_a_named_error_not_an_abort() -> None:
    """Same contract as every other flush stage: soft-fail, but say so."""

    class _Exploding(Channel):
        name = "exploding"

        async def populate(self, store: Any, session_id: str) -> None:
            raise RuntimeError("boom")

    mem, _ = _memory_with_fake_store([_Exploding()])

    result = await mem.flush_memory("s1")

    assert any("populate:exploding" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_flush_without_channels_populates_nothing() -> None:
    mem, _ = _memory_with_fake_store([])

    result = await mem.flush_memory("s1")

    assert not any("populate" in e for e in result["errors"])


def test_build_retriever_appends_caller_channels() -> None:
    from graphknows.retrieval import build_retriever
    from graphknows.settings import GraphKnowsSettings

    ch = _RecordingChannel()
    retriever = build_retriever(GraphKnowsSettings(), store=MagicMock(), extra_channels=[ch])

    assert ch in retriever._channels
