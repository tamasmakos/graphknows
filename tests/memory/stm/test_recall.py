"""Unit tests for recall_short_term from graphknows.memory.stm.recall."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.stm.recall import recall_short_term


def _embed(text: str) -> list[float]:
    return [0.1] * 10


def _make_mg(entity_chunks=None):
    mg = MagicMock()
    mg.query_entity_chunks = AsyncMock(return_value=entity_chunks or [])
    return mg


def _make_pg(chunks=None, turns=None):
    pg = MagicMock()
    pg.search_stm_chunks = AsyncMock(return_value=chunks or [])
    pg.search_stm_turns = AsyncMock(return_value=turns or [])
    return pg


def test_recall_returns_empty_when_no_results():
    mg = _make_mg()
    pg = _make_pg()
    results = recall_short_term(mg, pg, query="test", session_id="s1", embed_fn=_embed)
    assert results == []


def test_recall_returns_chunk_hits():
    chunk = {
        "id": "c1",
        "text": "hello world",
        "heading_path": "",
        "file_id": "f1",
        "metadata": {"memory_class": "fact", "entity_names": ["foo"]},
        "cosine_score": 0.9,
    }
    mg = _make_mg()
    pg = _make_pg(chunks=[chunk])
    results = recall_short_term(mg, pg, query="hello", session_id="s1", embed_fn=_embed)
    assert len(results) == 1
    assert results[0].text == "hello world"
    assert results[0].source_type == "chunk"


def test_recall_entity_boost_adds_to_score():
    chunk = {
        "id": "c1",
        "text": "alice in wonderland",
        "heading_path": "",
        "file_id": "f1",
        "metadata": {"memory_class": "episode", "entity_names": ["alice"]},
        "cosine_score": 0.7,
    }
    entity_chunk = {
        "chunk_id": "c1",
        "text": "alice in wonderland",
        "entity_names": ["alice"],
        "heading_path": "",
        "file_id": "f1",
        "memory_class": "episode",
        "metadata": {},
    }
    mg = _make_mg(entity_chunks=[entity_chunk])
    pg = _make_pg(chunks=[chunk])
    results = recall_short_term(mg, pg, query="alice visited", session_id="s1", embed_fn=_embed)
    assert len(results) == 1
    # Entity boosted chunk should have higher score than pure ANN
    assert results[0].score > 1.0 / (60 + 1 + 1)


def test_recall_respects_top_k():
    chunks = [
        {
            "id": f"c{i}",
            "text": f"chunk {i}",
            "heading_path": "",
            "file_id": "f1",
            "metadata": {"memory_class": "raw_evidence", "entity_names": []},
            "cosine_score": 0.5,
        }
        for i in range(10)
    ]
    mg = _make_mg()
    pg = _make_pg(chunks=chunks)
    results = recall_short_term(mg, pg, query="chunk", session_id="s1", top_k=3, embed_fn=_embed)
    assert len(results) <= 3


def test_recall_uses_default_embed_fn_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """recall_short_term should reach embed_one when embed_fn is None."""
    import sys
    import types

    called: list[str] = []

    # Intercept the lazy import inside recall_short_term by replacing the module
    fake_embedder = types.ModuleType("graphknows.memory.stores.embedder")

    def fake_embed_one(text: str) -> list[float]:
        called.append(text)
        return [0.0] * 3

    fake_embedder.embed_one = fake_embed_one  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "graphknows.memory.stores.embedder", fake_embedder)

    mg = _make_mg()
    pg = _make_pg()
    recall_short_term(mg, pg, query="hello", session_id="s1", embed_fn=None)
    assert called, "embed_one was not called — default embed_fn path not reached"
