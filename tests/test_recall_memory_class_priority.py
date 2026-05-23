"""Unit tests for memory_class-aware score shaping in recall."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.retrieval.recall_ltm import _shape_score as ltm_shape_score, recall_long_term
from graphknows.retrieval.recall_stm import _shape_score as stm_shape_score, recall_short_term


def _embed(text: str) -> list[float]:
    return [0.1] * 10


# --- _shape_score unit tests ---


def test_shape_score_fact_boosts():
    assert stm_shape_score(1.0, "fact") == 1.5
    assert ltm_shape_score(1.0, "fact") == 1.5


def test_shape_score_raw_evidence_deboosts():
    assert stm_shape_score(1.0, "raw_evidence") == 0.6
    assert ltm_shape_score(1.0, "raw_evidence") == 0.6


def test_shape_score_unknown_neutral():
    assert stm_shape_score(1.0, "unknown") == 1.0
    assert ltm_shape_score(1.0, "unknown") == 1.0


def test_shape_score_none_neutral():
    assert stm_shape_score(1.0, None) == 1.0
    assert ltm_shape_score(1.0, None) == 1.0


# --- STM recall integration tests ---


def test_stm_recall_distilled_outranks_raw():
    mg = MagicMock()
    mg.query_entity_chunks = AsyncMock(return_value=[])
    pg = MagicMock()
    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {
                "id": "raw1",
                "text": "raw text",
                "heading_path": "",
                "file_id": "f1",
                "metadata": {"memory_class": "raw_evidence"},
            },
            {
                "id": "fact1",
                "text": "fact text",
                "heading_path": "",
                "file_id": "f2",
                "metadata": {"memory_class": "fact"},
            },
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])

    results = recall_short_term(mg, pg, "test", session_id="s1", top_k=2, embed_fn=_embed)

    assert len(results) == 2
    # raw_evidence has higher base RRF (rank 0) but should be deboosted below fact (rank 1).
    assert results[0].text == "fact text"
    assert results[1].text == "raw text"


def test_shaping_preserves_top_k():
    mg = MagicMock()
    mg.query_entity_chunks = AsyncMock(return_value=[])
    pg = MagicMock()
    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {
                "id": f"c{i}",
                "text": f"text{i}",
                "heading_path": "",
                "file_id": "f",
                "metadata": {"memory_class": "fact"},
            }
            for i in range(20)
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])

    results = recall_short_term(mg, pg, "test", session_id="s1", top_k=5, embed_fn=_embed)
    assert len(results) <= 5


def test_shaping_handles_missing_metadata():
    mg = MagicMock()
    mg.query_entity_chunks = AsyncMock(return_value=[])
    pg = MagicMock()
    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {"id": "c1", "text": "no metadata", "heading_path": "", "file_id": "f1"},
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])

    results = recall_short_term(mg, pg, "test", session_id="s1", embed_fn=_embed)
    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0 / 61)


# --- LTM recall integration tests ---


def test_ltm_recall_distilled_outranks_raw():
    neo = MagicMock()
    neo._driver = MagicMock()
    neo._driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    pg = MagicMock()
    pg.search_ltm_chunks = AsyncMock(
        return_value=[
            {
                "id": "raw1",
                "text": "raw text",
                "heading_path": "",
                "session_id": "s1",
                "file_id": "f1",
                "metadata": {"memory_class": "raw_evidence"},
            },
            {
                "id": "fact1",
                "text": "fact text",
                "heading_path": "",
                "session_id": "s1",
                "file_id": "f2",
                "metadata": {"memory_class": "fact"},
            },
        ]
    )
    pg.search_ltm_turns = AsyncMock(return_value=[])

    results = recall_long_term(neo, pg, "test", top_k=2, embed_fn=_embed)

    assert len(results) == 2
    assert results[0].text == "fact text"
    assert results[1].text == "raw text"
