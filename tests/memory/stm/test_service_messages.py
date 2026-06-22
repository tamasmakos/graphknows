"""Unit tests for STMService.ingest_messages()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from graphknows.memory.stm.service import STMService


def _make_svc():
    mg = MagicMock()
    for attr in [
        "write_session_metadata",
        "write_dialog_node",
        "write_turn_node",
        "write_turn_sequence",
        "write_turn_entity_edge",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    pg = MagicMock()
    pg.write_stm_turn = AsyncMock(return_value=None)
    embed = MagicMock(return_value=[0.0] * 10)
    svc = STMService(mg, pg, extractor=None, embed_fn=embed)
    return svc, mg, pg


def _make_options(user_id="u1", agent_id="a1", run_id="r1", infer=True):
    from graphknows.memory.models.memory_class import MemoryClass
    from graphknows.memory.models.ingest_options import IngestOptions, MemoryScope

    return IngestOptions(
        scope=MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id),
        infer=infer,
        memory_class=MemoryClass.raw_evidence,
    )


def _make_messages(n=2):
    from graphknows.memory.models.message import Message, Role

    return [
        Message(role=Role.user if i % 2 == 0 else Role.assistant, content=f"message {i}")
        for i in range(n)
    ]


def test_ingest_messages_returns_ingest_result():
    from graphknows.memory.models.ingest import IngestResult

    svc, mg, pg = _make_svc()
    messages = _make_messages(3)
    options = _make_options()

    import asyncio
    from unittest.mock import patch

    def sync_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("graphknows.memory.stm._async_bridge._run", side_effect=sync_run):
        result = svc.ingest_messages(messages, options)

    assert isinstance(result, IngestResult)
    assert result.chunk_count == 3
    assert result.session_id != ""


def test_ingest_messages_chunk_count_matches_message_count():
    import asyncio
    from unittest.mock import patch

    svc, mg, pg = _make_svc()
    messages = _make_messages(5)
    options = _make_options()

    def sync_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("graphknows.memory.stm._async_bridge._run", side_effect=sync_run):
        result = svc.ingest_messages(messages, options)

    assert result.chunk_count == 5


def test_ingest_messages_no_infer_skips_extractor():
    import asyncio
    from unittest.mock import MagicMock, patch

    extractor = MagicMock()
    svc, mg, pg = _make_svc()
    svc._extractor = extractor
    messages = _make_messages(2)
    options = _make_options(infer=False)

    def sync_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("graphknows.memory.stm._async_bridge._run", side_effect=sync_run):
        svc.ingest_messages(messages, options)

    extractor.extract.assert_not_called()
