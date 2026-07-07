"""Unit tests for STMService — the first-class STM service layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.stm import STMConfig, STMService


def _make_service(extractor=None, embed_fn=None):
    stm = MagicMock()
    vec = MagicMock()
    for attr in [
        "write_session_metadata",
        "write_file_node",
        "write_chunk_node",
        "write_entity_node",
        "write_chunk_entity_edge",
        "write_dialog_node",
        "write_turn_node",
        "write_turn_sequence",
        "write_turn_entity_edge",
    ]:
        setattr(stm, attr, AsyncMock(return_value=None))
    vec.write_stm_chunk = AsyncMock(return_value=None)
    vec.write_stm_turn = AsyncMock(return_value=None)
    vec.search_stm_chunks = AsyncMock(return_value=[])
    vec.search_stm_turns = AsyncMock(return_value=[])
    stm.query_entity_chunks = AsyncMock(return_value=[])
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    svc = STMService(stm, vec, extractor=extractor, embed_fn=embed)
    return svc, stm, vec, embed


def test_stm_service_instantiates():
    svc, _, _, _ = _make_service()
    assert isinstance(svc, STMService)


def test_stm_service_has_config():
    svc, _, _, _ = _make_service()
    assert isinstance(svc._config, STMConfig)


def test_stm_service_accepts_custom_config():
    cfg = STMConfig(chunk_size=256, entity_confidence=0.8)
    stm = MagicMock()
    vec = MagicMock()
    embed = MagicMock(return_value=[0.0] * 10)
    svc = STMService(stm, vec, extractor=None, embed_fn=embed, config=cfg)
    assert svc._config.chunk_size == 256
    assert svc._config.entity_confidence == 0.8


@pytest.mark.asyncio
async def test_ingest_calls_handler(tmp_path):
    svc, stm, vec, embed = _make_service()
    result = await svc.ingest("Hello world", session_id="s1", title="Test")
    assert result.session_id == "s1"
    assert result.chunk_count >= 1


@pytest.mark.asyncio
async def test_record_turn_creates_dialog_capture():
    from graphknows.memory.stm.dialog import Turn

    svc, stm, vec, embed = _make_service()
    turn = Turn(id="t1", index=0, role="user", text="hello")
    result = await svc.record_turn("s1", "d1", turn)
    assert result.turn_id == "t1"
    assert svc._dialog_capture is not None


@pytest.mark.asyncio
async def test_ingest_handler_is_lazy():
    svc, _, _, _ = _make_service()
    assert svc._ingest_handler is None
    await svc.ingest("x", session_id="s1")
    assert svc._ingest_handler is not None
