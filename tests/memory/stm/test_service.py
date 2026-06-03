"""Unit tests for STMService — the first-class STM service layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.stm import STMConfig, STMService


def _make_service(extractor=None, embed_fn=None):
    mg = MagicMock()
    pg = MagicMock()
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
        setattr(mg, attr, AsyncMock(return_value=None))
    pg.write_stm_chunk = AsyncMock(return_value=None)
    pg.write_stm_turn = AsyncMock(return_value=None)
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    mg.query_entity_chunks = AsyncMock(return_value=[])
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    svc = STMService(mg, pg, extractor=extractor, embed_fn=embed)
    return svc, mg, pg, embed


def test_stm_service_instantiates():
    svc, _, _, _ = _make_service()
    assert isinstance(svc, STMService)


def test_stm_service_has_config():
    svc, _, _, _ = _make_service()
    assert isinstance(svc._config, STMConfig)


def test_stm_service_accepts_custom_config():
    cfg = STMConfig(chunk_size=256, entity_confidence=0.8)
    mg = MagicMock()
    pg = MagicMock()
    embed = MagicMock(return_value=[0.0] * 10)
    svc = STMService(mg, pg, extractor=None, embed_fn=embed, config=cfg)
    assert svc._config.chunk_size == 256
    assert svc._config.entity_confidence == 0.8


def test_ingest_calls_handler(tmp_path):
    svc, mg, pg, embed = _make_service()
    result = svc.ingest("Hello world", session_id="s1", title="Test")
    assert result.session_id == "s1"
    assert result.chunk_count >= 1


def test_recall_returns_list():
    svc, mg, pg, embed = _make_service()
    hits = svc.recall("test query", session_id="s1")
    assert isinstance(hits, list)


def test_record_turn_creates_dialog_capture():
    from graphknows.memory.stm.dialog import Turn

    svc, mg, pg, embed = _make_service()
    turn = Turn(id="t1", index=0, role="user", text="hello")
    result = svc.record_turn("s1", "d1", turn)
    assert result.turn_id == "t1"
    assert svc._dialog_capture is not None


def test_ingest_handler_is_lazy():
    svc, _, _, _ = _make_service()
    assert svc._ingest_handler is None
    svc.ingest("x", session_id="s1")
    assert svc._ingest_handler is not None
