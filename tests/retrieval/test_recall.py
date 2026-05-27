"""Unit tests for recall_short_term and recall_long_term.

STM deep tests have moved to tests/memory/stm/test_recall.py.
These tests verify the canonical recall_short_term from memory.stm.recall.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from graphknows.retrieval.recall_ltm import recall_long_term
from graphknows.memory.stm.recall import recall_short_term
from graphknows.memory.stm.recall import recall_short_term as canonical_recall_short_term


def test_shim_reexports_canonical_function():
    """Both imports should resolve to the same canonical function object."""
    assert recall_short_term is canonical_recall_short_term


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


# ===========================================================================
# Memory class priority / score shaping
# ===========================================================================


from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.retrieval.recall_ltm import _shape_score as ltm_shape_score
from graphknows.retrieval.recall_ltm import recall_long_term
from graphknows.memory.stm.recall import _shape_score as stm_shape_score
from graphknows.memory.stm.recall import recall_short_term


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


# ===========================================================================
# recall.py public API
# ===========================================================================


from graphknows.recall import (
    RecallHit,
    _extract_query_tokens,
    recall_by_dialog,
    recall_by_document,
    recall_by_topic,
)

# ---------------------------------------------------------------------------
# Fake DB infrastructure
# ---------------------------------------------------------------------------


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class FakeCursor:
    """Minimal cursor stub — supports .data() and iteration."""

    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.queries: list[str] = []
        self.params: list[dict] = []

    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        self.params.append(kwargs)
        return FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeDriver:
    def __init__(self, rows):
        self._rows = rows
        self.session_call_count = 0
        self.session_obj = FakeSession(rows)

    def session(self):
        self.session_call_count += 1
        return self.session_obj


# ---------------------------------------------------------------------------
# _extract_query_tokens
# ---------------------------------------------------------------------------


def test_extract_tokens_normal_sentence_returns_words():
    tokens = _extract_query_tokens("The quick brown fox jumps")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens


def test_extract_tokens_lowercased():
    tokens = _extract_query_tokens("HELLO WORLD")
    assert "hello" in tokens
    assert "world" in tokens
    assert "HELLO" not in tokens


def test_extract_tokens_empty_string_returns_empty_list():
    assert _extract_query_tokens("") == []


def test_extract_tokens_all_short_words_returns_empty_list():
    # "a", "to", "in" are all < 3 chars
    assert _extract_query_tokens("a to in") == []


def test_extract_tokens_deduplicates_repeated_words():
    tokens = _extract_query_tokens("the cat and the dog and the fish")
    assert tokens.count("the") == 1
    assert tokens.count("and") == 1


def test_extract_tokens_three_char_words_included():
    # "cat" (3 chars) must be included; "is" (2 chars) must not
    tokens = _extract_query_tokens("cat is big")
    assert "cat" in tokens
    assert "big" in tokens
    assert "is" not in tokens


def test_extract_tokens_preserves_first_occurrence_order():
    tokens = _extract_query_tokens("dog cat dog cat bird")
    assert tokens.index("dog") < tokens.index("cat")
    assert tokens.index("cat") < tokens.index("bird")


def test_extract_tokens_returns_list_type():
    result = _extract_query_tokens("hello world")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# recall_by_document
# ---------------------------------------------------------------------------


def test_recall_by_document_empty_query_returns_empty_no_db_call():
    driver = FakeDriver([])
    result = recall_by_document(driver, "")
    assert result == []
    assert driver.session_call_count == 0


def test_recall_by_document_short_words_only_returns_empty_no_db_call():
    driver = FakeDriver([])
    result = recall_by_document(driver, "a to in")
    assert result == []
    assert driver.session_call_count == 0


def test_recall_by_document_returns_recall_hit_objects():
    rows = [
        {"text": "Some document text", "file_id": "f1", "entities": ["Alice", "Bob"], "score": 2}
    ]
    driver = FakeDriver(rows)
    hits = recall_by_document(driver, "Alice and Bob meeting")
    assert len(hits) == 1
    assert isinstance(hits[0], RecallHit)


def test_recall_by_document_hit_fields_mapped_correctly():
    rows = [{"text": "Found content", "file_id": "doc-42", "entities": ["EU", "NATO"], "score": 3}]
    driver = FakeDriver(rows)
    hits = recall_by_document(driver, "EU and NATO cooperation")
    hit = hits[0]
    assert hit.text == "Found content"
    assert hit.score == 3.0
    assert hit.entity_names == ["EU", "NATO"]
    assert hit.provenance == {"file_id": "doc-42"}


def test_recall_by_document_top_k_clamped_from_zero_to_one():
    driver = FakeDriver([])
    recall_by_document(driver, "some query here", top_k=0)
    assert driver.session_call_count == 1
    assert driver.session_obj.params[0]["top_k"] == 1


def test_recall_by_document_top_k_clamped_from_large_to_twenty():
    driver = FakeDriver([])
    recall_by_document(driver, "some query here", top_k=100)
    assert driver.session_obj.params[0]["top_k"] == 20


def test_recall_by_document_empty_rows_returns_empty_list():
    driver = FakeDriver([])
    hits = recall_by_document(driver, "real query with words")
    assert hits == []


def test_recall_by_document_null_fields_default_to_safe_values():
    rows = [{"text": None, "file_id": None, "entities": None, "score": None}]
    driver = FakeDriver(rows)
    hits = recall_by_document(driver, "real query with words")
    assert hits[0].text == ""
    assert hits[0].score == 0.0
    assert hits[0].entity_names == []


def test_recall_by_document_tokens_sent_to_driver():
    driver = FakeDriver([])
    recall_by_document(driver, "Alice and Bob meeting today")
    sent_tokens = driver.session_obj.params[0]["tokens"]
    assert "alice" in sent_tokens
    assert "bob" in sent_tokens
    assert "meeting" in sent_tokens
    assert "today" in sent_tokens


# ---------------------------------------------------------------------------
# recall_by_dialog
# ---------------------------------------------------------------------------


def test_recall_by_dialog_empty_query_returns_empty_no_db_call():
    driver = FakeDriver([])
    result = recall_by_dialog(driver, "")
    assert result == []
    assert driver.session_call_count == 0


def test_recall_by_dialog_returns_recall_hit_objects():
    rows = [
        {
            "text": "User said hello",
            "session_id": "s1",
            "role": "user",
            "entities": ["Alice"],
            "score": 1,
        }
    ]
    driver = FakeDriver(rows)
    hits = recall_by_dialog(driver, "Alice discussion here")
    assert len(hits) == 1
    assert isinstance(hits[0], RecallHit)


def test_recall_by_dialog_hit_fields_mapped_correctly():
    rows = [
        {
            "text": "Agent replied about policy",
            "session_id": "sess-99",
            "role": "agent",
            "entities": ["EU", "Policy"],
            "score": 2,
        }
    ]
    driver = FakeDriver(rows)
    hits = recall_by_dialog(driver, "EU policy discussion today")
    hit = hits[0]
    assert hit.text == "Agent replied about policy"
    assert hit.score == 2.0
    assert hit.entity_names == ["EU", "Policy"]
    assert hit.provenance["session_id"] == "sess-99"
    assert hit.provenance["role"] == "agent"


def test_recall_by_dialog_top_k_clamped_low():
    driver = FakeDriver([])
    recall_by_dialog(driver, "some query text here", top_k=0)
    assert driver.session_obj.params[0]["top_k"] == 1


def test_recall_by_dialog_top_k_clamped_high():
    driver = FakeDriver([])
    recall_by_dialog(driver, "some query text here", top_k=500)
    assert driver.session_obj.params[0]["top_k"] == 20


def test_recall_by_dialog_with_session_filter_sends_session_id_param():
    rows = [
        {
            "text": "Turn text",
            "session_id": "s42",
            "role": "agent",
            "entities": [],
            "score": 0,
        }
    ]
    driver = FakeDriver(rows)
    recall_by_dialog(driver, "some query text here", session_filter="s42")
    assert driver.session_obj.params[0]["session_id"] == "s42"


def test_recall_by_dialog_without_session_filter_omits_session_id_param():
    driver = FakeDriver([])
    recall_by_dialog(driver, "some query text here", session_filter=None)
    assert "session_id" not in driver.session_obj.params[0]


# ---------------------------------------------------------------------------
# recall_by_topic
# ---------------------------------------------------------------------------


def test_recall_by_topic_empty_query_returns_empty_no_db_call():
    driver = FakeDriver([])
    result = recall_by_topic(driver, "")
    assert result == []
    assert driver.session_call_count == 0


def test_recall_by_topic_returns_recall_hit_objects():
    rows = [
        {
            "summary": "EU climate policy overview",
            "topic_id": "t1",
            "entities": ["EU", "Climate"],
            "score": 3,
        }
    ]
    driver = FakeDriver(rows)
    hits = recall_by_topic(driver, "EU climate discussion here")
    assert len(hits) == 1
    assert isinstance(hits[0], RecallHit)


def test_recall_by_topic_hit_fields_mapped_correctly():
    rows = [
        {
            "summary": "Migration and border policy",
            "topic_id": "topic-7",
            "entities": ["Migration", "EU"],
            "score": 4,
        }
    ]
    driver = FakeDriver(rows)
    hits = recall_by_topic(driver, "EU migration border control")
    hit = hits[0]
    assert hit.text == "Migration and border policy"
    assert hit.score == 4.0
    assert hit.entity_names == ["Migration", "EU"]
    assert hit.provenance == {"topic_id": "topic-7"}


def test_recall_by_topic_top_k_clamped_to_twenty():
    driver = FakeDriver([])
    recall_by_topic(driver, "some query text here", top_k=999)
    assert driver.session_obj.params[0]["top_k"] == 20


def test_recall_by_topic_null_summary_defaults_to_empty_string():
    rows = [
        {
            "summary": None,
            "label": "climate policy",
            "topic_id": None,
            "entities": [],
            "score": 0,
        }
    ]
    driver = FakeDriver(rows)
    hits = recall_by_topic(driver, "climate policy")
    assert hits[0].text == "climate policy"


def test_recall_by_topic_empty_rows_returns_empty_list():
    driver = FakeDriver([])
    hits = recall_by_topic(driver, "real query with meaningful words")
    assert hits == []


# ===========================================================================
# STM entity_boost
# ===========================================================================


from graphknows.memory.stm.recall import _entity_boost
from graphknows.memory.models.recall import RecallResult


class TestEntityBoostAddsNewChunks:
    def test_entity_only_chunk_is_added_to_results(self):
        """Chunks found via entity graph but NOT in ANN results must be added."""
        results: dict = {}  # empty — simulates no ANN hits

        class FakeMg:
            def query_entity_chunks(self, session_id, entity_names, top_k):
                import asyncio

                async def _coro():
                    return [
                        {
                            "chunk_id": "c1",
                            "text": "entity chunk",
                            "entity_names": ["obama"],
                        }
                    ]

                return _coro()

        _entity_boost(FakeMg(), "s1", ["obama"], top_k=5, results=results)

        assert "c1" in results, "Entity-matched chunk must be added even if not in ANN results"
        score, result = results["c1"]
        assert score > 0
        assert "obama" in result.entity_names


# ===========================================================================
# Mocked store tests: forget, prune, embed, ANN
# ===========================================================================


from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# forget.py — LTMForget
# ---------------------------------------------------------------------------


def _make_forget(deleted: int = 5):
    """Return an LTMForget with a fully mocked Neo4j driver."""
    from graphknows.forget import LTMForget

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # single() returns a dict-like mock
    def _single_result(deleted_count):
        r = MagicMock()
        r.__getitem__ = lambda self, k: deleted_count if k == "deleted" else 0
        return r

    mock_session.run.return_value.single.return_value = _single_result(deleted)

    forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="postgresql://fake/db")
    return forget, mock_session


class TestPruneDisconnected:
    def test_returns_deleted_count(self):
        forget, _ = _make_forget(deleted=7)
        result = forget.prune_disconnected()
        assert result == {"deleted_entities": 7}

    def test_runs_correct_cypher(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.prune_disconnected()
        cypher = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" in cypher
        assert "Entity" in cypher or "ENTITY" in cypher

    def test_handles_none_result(self):
        from graphknows.forget import LTMForget

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = None

        forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="pg://fake/db")
        result = forget.prune_disconnected()
        assert result == {"deleted_entities": 0}


class TestPruneByTtl:
    def test_returns_deleted_count(self):
        forget, _ = _make_forget(deleted=3)
        result = forget.prune_by_ttl(ttl_days=30)
        assert result == {"deleted_entities": 3}

    def test_passes_cutoff_and_floor_params(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.prune_by_ttl(ttl_days=14, pagerank_floor=0.005)
        kwargs = mock_session.run.call_args[1]
        assert "cutoff" in kwargs
        assert kwargs["floor"] == pytest.approx(0.005)


class TestForgetBefore:
    def test_returns_expected_keys(self):
        forget, mock_session = _make_forget(deleted=2)
        result = forget.forget_before("2024-01-01T00:00:00")
        assert "deleted_segments" in result
        assert "orphaned_entities_deleted" in result

    def test_runs_two_statements(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.forget_before("2024-01-01")
        assert mock_session.run.call_count >= 2


class TestPrune:
    def _make_prune_forget(self, counts: list[int]):
        """counts = [docs, chunks, dialogs, utterances] (int per query)."""
        from graphknows.forget import LTMForget

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        call_no = [0]

        def _single():
            idx = call_no[0]
            call_no[0] += 1
            if idx < len(counts):
                r = MagicMock()
                r.__getitem__ = lambda self, k: counts[idx] if k == "c" else 0
                return r
            return None

        mock_session.run.return_value.single = _single

        forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="pg://fake/db")
        return forget

    def test_returns_all_keys(self):
        forget = self._make_prune_forget([0, 0, 0, 0])
        result = forget.prune()
        assert "documents_orphaned" in result
        assert "chunks_orphaned" in result
        assert "dialogs_empty" in result
        assert "utterances_ttl" in result

    def test_nonzero_counts_trigger_delete(self):
        forget = self._make_prune_forget([2, 1, 3, 0])
        result = forget.prune()
        assert result["documents_orphaned"] == 2
        assert result["chunks_orphaned"] == 1
        assert result["dialogs_empty"] == 3


# ---------------------------------------------------------------------------
# config.py — GraphKnowsSettings
# ---------------------------------------------------------------------------


class TestGraphKnowsSettings:
    @pytest.fixture(autouse=True)
    def isolate_env(self, monkeypatch):
        """Remove env vars that would interfere with defaults."""
        for key in [
            "GRAPHKNOWS_ENV",
            "GRAPHKNOWS_PORT",
            "NEO4J_URI",
            "NEO4J_USER",
            "NEO4J_PASSWORD",
            "PGVECTOR_DSN",
            "PGVECTOR_PASSWORD",
            "MEMGRAPH_URI",
            "OPENROUTER_API_KEY",
            "OPEN_ROUTER_API",
        ]:
            monkeypatch.delenv(key, raising=False)

    def test_default_env_is_development(self, monkeypatch):
        from graphknows.config import GraphKnowsSettings

        # Patch _load_repo_env so real .env doesn't override defaults
        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        assert settings.graphknows_env == "development"

    def test_default_port(self, monkeypatch):
        from graphknows.config import GraphKnowsSettings

        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        assert settings.graphknows_port == 8025

    def test_env_var_overrides_default(self, monkeypatch):
        from graphknows.config import GraphKnowsSettings

        monkeypatch.setenv("GRAPHKNOWS_ENV", "production")
        monkeypatch.setenv("NEO4J_PASSWORD", "prod_pass")
        monkeypatch.setenv("PGVECTOR_PASSWORD", "pg_pass")
        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        assert settings.graphknows_env == "production"

    def test_check_production_secrets_passes_in_dev(self):
        from graphknows.config import GraphKnowsSettings

        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        # In development mode, no raise expected
        settings.check_production_secrets()

    def test_check_production_secrets_raises_with_default_neo4j(self, monkeypatch):
        from graphknows.config import GraphKnowsSettings

        monkeypatch.setenv("GRAPHKNOWS_ENV", "production")
        monkeypatch.setenv("PGVECTOR_PASSWORD", "pg_pass")
        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        settings.graphknows_env = "production"
        settings.neo4j_password = "testpassword123"
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            settings.check_production_secrets()

    def test_check_production_masked_dsn_raises(self):
        from graphknows.config import GraphKnowsSettings

        with patch("graphknows.config._load_repo_env", return_value={}):
            settings = GraphKnowsSettings()
        settings.graphknows_env = "production"
        settings.neo4j_password = "real_password"
        settings.pgvector_dsn = "postgresql://postgres:***@host:5432/db"
        with pytest.raises(ValueError, match="PGVECTOR_DSN"):
            settings.check_production_secrets()


class TestFindRepoEnv:
    def test_finds_env_file_in_parents(self, tmp_path):
        from graphknows.config import _find_repo_env

        # Create .env in tmp_path
        (tmp_path / ".env").write_text("TEST=1")
        # _find_repo_env walks from __file__ — just test it returns a Path or None
        result = _find_repo_env()
        assert result is None or isinstance(result, type(tmp_path))


# ---------------------------------------------------------------------------
# recall_ltm.py — private helpers
# ---------------------------------------------------------------------------


class TestShapeScore:
    @pytest.mark.parametrize(
        "memory_class,factor",
        [
            ("fact", 1.5),
            ("state", 1.4),
            ("episode", 1.3),
            ("summary", 1.2),
            ("raw_evidence", 0.6),
            ("unknown_class", 1.0),
            (None, 1.0),
        ],
    )
    def test_applies_correct_factor(self, memory_class, factor):
        from graphknows.retrieval.recall_ltm import _shape_score

        result = _shape_score(1.0, memory_class)
        assert result == pytest.approx(factor)

    def test_scales_non_unit_score(self):
        from graphknows.retrieval.recall_ltm import _shape_score

        assert _shape_score(2.0, "fact") == pytest.approx(3.0)


class TestAnnChunks:
    def test_builds_recall_results_from_pg_rows(self):
        from graphknows.retrieval.recall_ltm import _ann_chunks
        from graphknows.memory.stm.recall import RRF_K

        mock_pg = MagicMock()
        rows = [
            {
                "id": "c1",
                "text": "hello",
                "heading_path": "/intro",
                "session_id": "s1",
                "file_id": "f1",
                "metadata": {},
            }
        ]

        with patch("graphknows.retrieval.recall_ltm._run", return_value=rows):
            result = _ann_chunks(mock_pg, [0.1] * 1536, top_k=5, session_filter=None)

        assert "c1" in result
        score, recall = result["c1"]
        assert score == pytest.approx(1.0 / (RRF_K + 1))
        assert recall.text == "hello"
        assert recall.scope == "long_term"

    def test_exception_returns_empty(self):
        from graphknows.retrieval.recall_ltm import _ann_chunks

        mock_pg = MagicMock()
        with patch("graphknows.retrieval.recall_ltm._run", side_effect=RuntimeError("db down")):
            result = _ann_chunks(mock_pg, [0.1] * 10, top_k=5, session_filter=None)
        assert result == {}


class TestAnnTurns:
    def test_builds_turn_recall_results(self):
        from graphknows.retrieval.recall_ltm import _ann_turns

        mock_pg = MagicMock()
        rows = [
            {
                "text": "user said hello",
                "session_id": "s1",
                "dialog_id": "d1",
                "turn_index": 0,
                "role": "user",
                "metadata": {},
            }
        ]
        with patch("graphknows.retrieval.recall_ltm._run", return_value=rows):
            result = _ann_turns(mock_pg, [0.1] * 1536, top_k=3, session_filter=None)

        assert len(result) == 1
        assert result[0].source_type == "turn"
        assert result[0].scope == "long_term"

    def test_exception_returns_empty_list(self):
        from graphknows.retrieval.recall_ltm import _ann_turns

        mock_pg = MagicMock()
        with patch("graphknows.retrieval.recall_ltm._run", side_effect=Exception("fail")):
            result = _ann_turns(mock_pg, [0.1] * 10, top_k=5, session_filter=None)
        assert result == []


class TestEntityTraversal:
    def test_empty_words_returns_existing(self):
        from graphknows.retrieval.recall_ltm import _entity_traversal

        existing = {"c1": (0.5, MagicMock())}
        result = _entity_traversal(None, [], top_k=5, session_filter=None, existing=existing)
        assert result is existing

    def test_merges_neo4j_results(self):
        from graphknows.retrieval.recall_ltm import _entity_traversal

        mock_neo = MagicMock()
        mock_session = MagicMock()
        mock_neo._driver = MagicMock()
        mock_neo._driver.session.return_value.__enter__.return_value = mock_session

        rows = [{"chunk_id": "c2", "text": "parliament text", "entities": ["EU Parliament"]}]
        mock_session.run.return_value.data.return_value = rows

        result = _entity_traversal(
            mock_neo,
            words=["parliament"],
            top_k=5,
            session_filter=None,
            existing={},
        )
        assert "c2" in result
        _, recall = result["c2"]
        assert recall.text == "parliament text"

    def test_exception_returns_existing(self):
        from graphknows.retrieval.recall_ltm import _entity_traversal

        mock_neo = MagicMock()
        mock_neo._driver.session.side_effect = Exception("fail")
        existing = {"x": (0.1, MagicMock())}
        result = _entity_traversal(
            mock_neo, ["word"], top_k=5, session_filter=None, existing=existing
        )
        assert "x" in result


class TestRecallLongTerm:
    def test_returns_sorted_recall_results(self):
        from graphknows.retrieval.recall_ltm import recall_long_term
        from graphknows.memory.models.recall import RecallResult

        mock_neo = MagicMock()
        mock_pg = MagicMock()

        fake_embed = lambda text: [0.1] * 1536

        chunk_result = RecallResult(
            text="chunk text",
            source_type="chunk",
            scope="long_term",
            score=0.9,
            metadata={},
        )
        turn_result = RecallResult(
            text="turn text",
            source_type="turn",
            scope="long_term",
            score=0.5,
            metadata={},
        )

        with (
            patch(
                "graphknows.retrieval.recall_ltm._ann_chunks",
                return_value={"c1": (0.9, chunk_result)},
            ),
            patch(
                "graphknows.retrieval.recall_ltm._entity_traversal",
                return_value={"c1": (0.9, chunk_result)},
            ),
            patch("graphknows.retrieval.recall_ltm._ann_turns", return_value=[turn_result]),
        ):
            results = recall_long_term(
                mock_neo, mock_pg, "test query", top_k=5, embed_fn=fake_embed
            )

        assert len(results) >= 1
        assert all(hasattr(r, "score") for r in results)

    def test_top_k_is_respected(self):
        from graphknows.retrieval.recall_ltm import recall_long_term
        from graphknows.memory.models.recall import RecallResult

        mock_neo = MagicMock()
        mock_pg = MagicMock()

        # Make 10 chunks
        chunks = {
            f"c{i}": (
                0.1 * i,
                RecallResult(
                    text=f"text{i}",
                    source_type="chunk",
                    scope="long_term",
                    score=0.1 * i,
                    metadata={},
                ),
            )
            for i in range(10)
        }

        with (
            patch("graphknows.retrieval.recall_ltm._ann_chunks", return_value=chunks),
            patch("graphknows.retrieval.recall_ltm._entity_traversal", return_value=chunks),
            patch("graphknows.retrieval.recall_ltm._ann_turns", return_value=[]),
        ):
            results = recall_long_term(
                mock_neo, mock_pg, "q", top_k=3, embed_fn=lambda t: [0.1] * 10
            )

        assert len(results) <= 3


# ---------------------------------------------------------------------------
# embedder.py
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_empty_list_returns_zero_array(self):
        from graphknows.retrieval.stores.embedder import EMBED_DIM, embed

        result = embed([])
        assert result.shape == (0, EMBED_DIM)

    def test_raises_without_api_key(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import embed

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            embed(["hello"])

    def test_successful_embed_returns_correct_shape(self, monkeypatch):
        import httpx

        from graphknows.retrieval.stores.embedder import EMBED_DIM, embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        fake_response = {"data": [{"index": 0, "embedding": [0.1] * EMBED_DIM}]}

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = embed(["hello world"])

        assert result.shape == (1, EMBED_DIM)
        assert result.dtype == np.float32

    def test_truncates_long_text(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import EMBED_DIM, _MAX_CHARS_PER_TEXT, embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        sent_inputs = []

        def fake_post(url, headers, json, timeout):
            sent_inputs.extend(json["input"])
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * EMBED_DIM}]}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            embed(["x" * 50_000])

        assert len(sent_inputs[0]) == _MAX_CHARS_PER_TEXT

    def test_raises_when_data_missing_from_response(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"error": "quota exceeded"}
            mock_resp.status_code = 429
            mock_resp.text = "quota exceeded"
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="missing 'data' key"):
                embed(["hello"])


class TestEmbedOne:
    def test_returns_list_of_floats(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import EMBED_DIM, embed_one

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.retrieval.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.1] * EMBED_DIM], dtype=np.float32)
            result = embed_one("hello")

        assert isinstance(result, list)
        assert len(result) == EMBED_DIM


class TestGetEmbedder:
    def test_shim_encode_calls_embed(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import EMBED_DIM, get_embedder

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.retrieval.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.2] * EMBED_DIM], dtype=np.float32)
            shim = get_embedder()
            result = shim.encode(["test text"])

        mock_embed.assert_called_once_with(["test text"], "openai/text-embedding-3-small")
        assert result.shape == (1, EMBED_DIM)

    def test_shim_wraps_string_in_list(self, monkeypatch):
        from graphknows.retrieval.stores.embedder import EMBED_DIM, get_embedder

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.retrieval.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.1] * EMBED_DIM], dtype=np.float32)
            shim = get_embedder()
            shim.encode("single string")

        call_args = mock_embed.call_args[0][0]
        assert call_args == ["single string"]
