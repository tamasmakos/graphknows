"""Unit tests for STMService — the first-class STM service layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.stm import STMConfig, STMService


def _make_store():
    store = MagicMock()
    for attr in [
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_relation",
        "write_temporal",
        "connect",
        "ensure_schema",
        "close",
    ]:
        setattr(store, attr, AsyncMock(return_value=None))
    return store


def _make_service(extractor=None, embed_fn=None):
    store = _make_store()
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    svc = STMService(store, extractor=extractor, embed_fn=embed)
    return svc, store, embed


def test_stm_service_instantiates():
    svc, _, _ = _make_service()
    assert isinstance(svc, STMService)


def test_stm_service_has_config():
    svc, _, _ = _make_service()
    assert isinstance(svc._config, STMConfig)


def test_stm_service_accepts_custom_config():
    cfg = STMConfig(chunk_size=256, entity_confidence=0.8)
    store = _make_store()
    embed = MagicMock(return_value=[0.0] * 10)
    svc = STMService(store, extractor=None, embed_fn=embed, config=cfg)
    assert svc._config.chunk_size == 256
    assert svc._config.entity_confidence == 0.8


@pytest.mark.asyncio
async def test_ingest_calls_handler(tmp_path):
    svc, store, embed = _make_service()
    result = await svc.ingest("Hello world", session_id="s1", title="Test")
    assert result.session_id == "s1"
    assert result.chunk_count >= 1
    store.write_chunk.assert_awaited()


@pytest.mark.asyncio
async def test_record_turn_writes_turn_chunk():
    from graphknows.ingestion.stm.dialog import Turn

    svc, store, embed = _make_service()
    turn = Turn(id="t1", index=0, role="user", text="hello")
    result = await svc.record_turn("s1", "d1", turn)
    assert result.turn_id == "t1"
    store.write_chunk.assert_awaited()
    # The turn is written as a CHUNK with kind='turn'.
    written = store.write_chunk.await_args.args[0]
    assert written["kind"] == "turn"


@pytest.mark.asyncio
async def test_ingest_handler_is_lazy():
    svc, _, _ = _make_service()
    assert svc._ingest_handler is None
    await svc.ingest("x", session_id="s1")
    assert svc._ingest_handler is not None
