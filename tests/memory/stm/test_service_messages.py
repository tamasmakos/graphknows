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
    assert result.chunk_count == 3
    assert result.session_id != ""


async def test_ingest_messages_chunk_count_matches_message_count():
    svc, store = _make_svc()
    messages = _make_messages(5)
    options = _make_options()

    result = await svc.ingest_messages(messages, options)

    assert result.chunk_count == 5


async def test_ingest_messages_no_infer_skips_extractor():
    extractor = MagicMock()
    svc, store = _make_svc()
    svc._extractor = extractor
    messages = _make_messages(2)
    options = _make_options(infer=False)

    await svc.ingest_messages(messages, options)

    extractor.extract.assert_not_called()
