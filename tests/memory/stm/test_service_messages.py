"""Unit tests for STMService.ingest_messages()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.stm.service import STMService

pytestmark = pytest.mark.asyncio


def _make_svc():
    store = MagicMock()
    for attr in [
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_relation",
        "write_temporal",
    ]:
        setattr(store, attr, AsyncMock(return_value=None))
    embed = MagicMock(return_value=[0.0] * 10)
    svc = STMService(store, extractor=None, embed_fn=embed)
    return svc, store


def _make_options(user_id="u1", agent_id="a1", run_id="r1", infer=True):
    from graphknows.models.ingest_options import IngestOptions, MemoryScope
    from graphknows.models.memory_class import MemoryClass

    return IngestOptions(
        scope=MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id),
        infer=infer,
        memory_class=MemoryClass.raw_evidence,
    )


def _make_messages(n=2):
    from graphknows.models.message import Message, Role

    return [
        Message(role=Role.user if i % 2 == 0 else Role.assistant, content=f"message {i}")
        for i in range(n)
    ]


async def test_ingest_messages_returns_ingest_result():
    from graphknows.models.ingest import IngestResult

    svc, store = _make_svc()
    messages = _make_messages(3)
    options = _make_options()

    result = await svc.ingest_messages(messages, options)

    assert isinstance(result, IngestResult)
    # 3 turns is exactly one stride, so one window is emitted.
    assert result.chunk_count == 1
    assert result.session_id != ""


async def test_chunk_count_is_chunks_written_not_messages_received():
    """The count must follow the writes, not the input length.

    Turns are grouped into sliding windows, so reporting ``len(messages)`` would
    claim chunks that were never written — and every caller (flush, the eval
    harness) treats that number as the graph's size.
    """
    svc, store = _make_svc()

    result = await svc.ingest_messages(_make_messages(5), _make_options())

    assert result.chunk_count == store.write_chunk.await_count


async def test_window_chunking_off_restores_one_chunk_per_turn(monkeypatch):
    """The A/B control arm must be an exact replay, not an approximation."""
    monkeypatch.setenv("GRAPHKNOWS_WINDOW_CHUNKS", "false")
    svc, store = _make_svc()

    result = await svc.ingest_messages(_make_messages(5), _make_options())

    assert result.chunk_count == 5
    assert store.write_chunk.await_count == 5


class TestWindowChunking:
    """A retrievable CHUNK is a 5-turn sliding window, not a single turn.

    A lone turn is a poor retrieval unit — "She said yes" carries nothing an
    embedding can match. The extractor has always read a 5-turn window; this
    makes the STORED chunk that same window, so what is retrieved carries the
    context the extractor already relied on. Windows overlap (stride 3) so no
    adjacent pair of turns is ever split across a boundary.
    """

    async def test_turns_are_grouped_into_overlapping_windows(self):
        svc, store = _make_svc()
        result = await svc.ingest_messages(_make_messages(9), _make_options())
        # 9 turns at stride 3 -> windows ending at turns 3, 6, 9.
        assert result.chunk_count == 3
        assert store.write_chunk.await_count == 3

    async def test_a_chunk_carries_the_whole_window_not_one_turn(self):
        svc, store = _make_svc()
        await svc.ingest_messages(_make_messages(6), _make_options())
        texts = [
            c.kwargs.get("text") or c.args[0]["text"] for c in store.write_chunk.await_args_list
        ]
        # The second window covers turns 2..6, so it must contain more than its
        # own last turn — that context is the entire point.
        assert any("message 1" in t and "message 5" in t for t in texts), texts

    async def test_trailing_turns_are_not_lost(self):
        """7 turns at stride 3 leaves one past the last full window."""
        svc, store = _make_svc()
        result = await svc.ingest_messages(_make_messages(7), _make_options())
        texts = [
            c.kwargs.get("text") or c.args[0]["text"] for c in store.write_chunk.await_args_list
        ]
        assert any("message 6" in t for t in texts), texts
        assert result.chunk_count == store.write_chunk.await_count

    async def test_every_turn_reaches_some_chunk(self):
        svc, store = _make_svc()
        await svc.ingest_messages(_make_messages(11), _make_options())
        joined = " || ".join(
            c.kwargs.get("text") or c.args[0]["text"] for c in store.write_chunk.await_args_list
        )
        missing = [i for i in range(11) if f"message {i}" not in joined]
        assert not missing, f"turns absent from every chunk: {missing}"


async def test_ingest_messages_no_infer_skips_extractor():
    extractor = MagicMock()
    svc, store = _make_svc()
    svc._extractor = extractor
    messages = _make_messages(2)
    options = _make_options(infer=False)

    await svc.ingest_messages(messages, options)

    extractor.extract.assert_not_called()
