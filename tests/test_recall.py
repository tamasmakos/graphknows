"""Unit tests for recall_short_term and recall_long_term."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.retrieval.recall_stm import RecallResult, recall_short_term
from graphknows.retrieval.recall_ltm import recall_long_term


def _embed(text: str) -> list[float]:
    return [0.1] * 10


def _make_mg():
    mg = MagicMock()
    mg.query_entity_chunks = AsyncMock(return_value=[])
    return mg


def _make_pg(chunks=None, turns=None):
    pg = MagicMock()
    pg.search_stm_chunks = AsyncMock(return_value=chunks or [])
    pg.search_stm_turns = AsyncMock(return_value=turns or [])
    pg.search_ltm_chunks = AsyncMock(return_value=chunks or [])
    return pg


# --- STM recall ---


def test_recall_short_term_calls_search_stm_chunks():
    mg = _make_mg()
    pg = _make_pg(chunks=[{"id": "c1", "text": "hello", "heading_path": "", "file_id": "f1"}])
    results = recall_short_term(mg, pg, "hello world", session_id="s1", embed_fn=_embed)
    pg.search_stm_chunks.assert_called_once()
    assert any(r.source_type == "chunk" for r in results)


def test_recall_short_term_calls_search_stm_turns():
    mg = _make_mg()
    pg = _make_pg(turns=[{"id": "t1", "text": "agent said hi", "role": "agent"}])
    results = recall_short_term(mg, pg, "hello", session_id="s1", embed_fn=_embed)
    pg.search_stm_turns.assert_called_once()


def test_recall_short_term_calls_entity_chunks_in_memgraph():
    mg = _make_mg()
    mg.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "c1", "text": "EU doc", "entity_names": ["EU"]}]
    )
    pg = _make_pg(chunks=[{"id": "c1", "text": "EU doc", "heading_path": "", "file_id": "f1"}])
    results = recall_short_term(mg, pg, "EU policy", session_id="s1", embed_fn=_embed)
    mg.query_entity_chunks.assert_called_once()
    # c1 should be boosted
    assert len(results) >= 1
    eu_results = [r for r in results if "EU" in r.entity_names]
    assert len(eu_results) >= 1


def test_entity_found_chunk_gets_boosted_score():
    mg = _make_mg()
    mg.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "c1", "text": "X", "entity_names": ["EU"]}]
    )
    # Both ANN and entity path return c1
    pg = _make_pg(chunks=[{"id": "c1", "text": "EU text", "heading_path": "", "file_id": "f1"}])
    results = recall_short_term(mg, pg, "EU deal", session_id="s1", embed_fn=_embed)
    c1_results = [r for r in results if r.text == "EU text"]
    # Score should be sum of ANN + entity RRF
    assert len(c1_results) == 1
    assert c1_results[0].score > 1.0 / (60 + 1)  # boosted above base ANN score


def test_ann_failure_logs_warning_and_continues(caplog):
    mg = _make_mg()
    pg = _make_pg()
    pg.search_stm_chunks = AsyncMock(side_effect=RuntimeError("DB down"))
    pg.search_stm_turns = AsyncMock(return_value=[{"id": "t1", "text": "hello", "role": "user"}])
    with caplog.at_level("WARNING"):
        results = recall_short_term(mg, pg, "hello", session_id="s1", embed_fn=_embed)
    assert "stm_chunks ANN failed" in caplog.text
    assert len(results) >= 1  # turns still returned


def test_results_sorted_by_score_descending():
    mg = _make_mg()
    chunks = [
        {"id": f"c{i}", "text": f"text{i}", "heading_path": "", "file_id": "f"} for i in range(3)
    ]
    pg = _make_pg(chunks=chunks)
    results = recall_short_term(mg, pg, "test", session_id="s1", embed_fn=_embed)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_respected():
    mg = _make_mg()
    chunks = [
        {"id": f"c{i}", "text": f"text{i}", "heading_path": "", "file_id": "f"} for i in range(20)
    ]
    pg = _make_pg(chunks=chunks)
    results = recall_short_term(mg, pg, "test", session_id="s1", top_k=3, embed_fn=_embed)
    assert len(results) <= 3


# --- LTM recall ---


def test_recall_long_term_calls_search_ltm_chunks():
    neo = MagicMock()
    neo._driver = MagicMock()
    neo._driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    pg = _make_pg(
        chunks=[
            {
                "id": "c1",
                "text": "ltm content",
                "heading_path": "",
                "session_id": "s1",
                "file_id": "f1",
            }
        ]
    )
    results = recall_long_term(neo, pg, "test query", embed_fn=_embed)
    pg.search_ltm_chunks.assert_called_once()


def test_recall_long_term_scope_is_long_term():
    neo = MagicMock()
    neo._driver = MagicMock()
    neo._driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    pg = _make_pg(
        chunks=[
            {
                "id": "c1",
                "text": "ltm content",
                "heading_path": "",
                "session_id": "s1",
                "file_id": "f1",
            }
        ]
    )
    results = recall_long_term(neo, pg, "test query", embed_fn=_embed)
    for r in results:
        assert r.scope == "long_term"


def test_recall_long_term_results_sorted_descending():
    neo = MagicMock()
    neo._driver = MagicMock()
    neo._driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    chunks = [
        {"id": f"c{i}", "text": f"text{i}", "heading_path": "", "session_id": "s1", "file_id": "f"}
        for i in range(5)
    ]
    pg = _make_pg(chunks=chunks)
    results = recall_long_term(neo, pg, "test", embed_fn=_embed)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
