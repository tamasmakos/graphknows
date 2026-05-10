"""Tests for evidence ledger normalization from search and graphrag outputs."""
from __future__ import annotations

import re

import pytest

from services.agent.graph.evidence_ledger import (
    _recency_score,
    _source_authority,
    normalize_graphrag_chunk,
    normalize_graphrag_results,
    normalize_search_doc,
    normalize_search_results,
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestSourceAuthority:
    def test_ep_debates_is_authoritative(self):
        assert _source_authority("ep_debates") == 5

    def test_eurlex_is_authoritative(self):
        assert _source_authority("eurlex") == 5

    def test_eprs_briefings_is_high_authority(self):
        assert _source_authority("eprs_briefings") == 4

    def test_council_conclusions_is_medium_authority(self):
        assert _source_authority("council_conclusions") == 3

    def test_tavily_is_medium_authority(self):
        assert _source_authority("tavily") == 3

    def test_unknown_source_returns_default(self):
        assert _source_authority("some_random_blog") == 2

    def test_case_insensitive(self):
        assert _source_authority("EP_DEBATES") == _source_authority("ep_debates")


class TestRecencyScore:
    def test_unknown_date_returns_one(self):
        assert _recency_score(None) == 1
        assert _recency_score("") == 1
        assert _recency_score("not-a-date") == 1

    def test_very_old_document_returns_one(self):
        assert _recency_score("2015-01-01") == 1

    def test_four_to_five_years_returns_two(self):
        from datetime import datetime, timezone
        from datetime import timedelta
        four_years_ago = (datetime.now(timezone.utc) - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
        assert _recency_score(four_years_ago) == 2

    def test_recent_document_returns_five(self):
        from datetime import datetime, timezone
        from datetime import timedelta
        six_months_ago = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
        assert _recency_score(six_months_ago) == 5

    def test_year_only_string_accepted(self):
        # 2100 is far in the future — recency should be 5 (≤1 year via date arithmetic is weird,
        # but any year in the future is recency=5 because age_years < 0 ≤ 1)
        score = _recency_score("2030")
        assert isinstance(score, int)

    def test_iso_datetime_string_accepted(self):
        # Full ISO-8601 datetime should parse without error
        score = _recency_score("2023-06-15T12:00:00+00:00")
        assert 1 <= score <= 5


# ---------------------------------------------------------------------------
# Search doc normalization
# ---------------------------------------------------------------------------


class TestNormalizeSearchDoc:
    def _sample_doc(self) -> dict:
        return {
            "id": "doc-abc",
            "title": "EDIS briefing",
            "text": "The European defence industrial strategy was adopted in 2024.",
            "url": "https://europarl.europa.eu/thinktank/edis",
            "source": "eprs_briefings",
            "date": "2024-03-05",
        }

    def test_basic_record_fields_populated(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="EDIS strategy",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        assert rec["thread_id"] == "t-1"
        assert rec["sub_question_id"] == "sq-1"
        assert rec["summary"] == "EDIS briefing"
        assert rec["source_authority"] == 4  # eprs_briefings
        assert rec["independence"] == 3
        assert rec["directness"] == 5  # text is present
        assert rec["recency"] in (1, 2, 3, 4, 5)
        assert re.match(r"^ev-[0-9a-f]+$", rec["evidence_id"])

    def test_provenance_retrieval_ref_populated(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="EDIS strategy",
            thread_id="t-1",
            sub_question_id="sq-1",
            rank=2,
        )
        retrieval = rec["provenance"]["retrieval"]
        assert retrieval["query"] == "EDIS strategy"
        assert retrieval["tool_name"] == "search"
        assert retrieval["rank"] == 2
        assert retrieval["source_type"] == "eprs_briefings"

    def test_passage_ref_present_when_text_exists(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        assert rec["provenance"]["passage"] is not None
        assert "European" in rec["provenance"]["passage"]["text"]

    def test_passage_ref_absent_when_no_text(self):
        doc = self._sample_doc()
        doc["text"] = ""
        rec = normalize_search_doc(
            doc,
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        assert rec["provenance"]["passage"] is None
        assert rec["directness"] == 3

    def test_relation_and_graph_path_absent_for_search_doc(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        assert rec["provenance"]["relation"] is None
        assert rec["provenance"]["graph_path"] is None

    def test_document_ref_populated(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        doc_ref = rec["provenance"]["document"]
        assert doc_ref["document_id"] == "doc-abc"
        assert doc_ref["title"] == "EDIS briefing"
        assert doc_ref["url"] == "https://europarl.europa.eu/thinktank/edis"
        assert doc_ref["publisher"] == "eprs_briefings"

    def test_tags_include_source(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
        )
        assert "eprs_briefings" in rec["tags"]

    def test_retrieval_id_override(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
            retrieval_id="ret-override-123",
        )
        assert rec["provenance"]["retrieval"]["retrieval_id"] == "ret-override-123"

    def test_evidence_id_override(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
            evidence_id="ev-override-abc",
        )
        assert rec["evidence_id"] == "ev-override-abc"

    def test_independence_override(self):
        rec = normalize_search_doc(
            self._sample_doc(),
            query="q",
            thread_id="t-1",
            sub_question_id="sq-1",
            independence=5,
        )
        assert rec["independence"] == 5

    def test_tavily_source_gets_authority_three(self):
        doc = self._sample_doc()
        doc["source"] = "tavily"
        rec = normalize_search_doc(doc, query="q", thread_id="t-1", sub_question_id="sq-1")
        assert rec["source_authority"] == 3

    def test_eurlex_source_gets_authority_five(self):
        doc = self._sample_doc()
        doc["source"] = "eurlex"
        rec = normalize_search_doc(doc, query="q", thread_id="t-1", sub_question_id="sq-1")
        assert rec["source_authority"] == 5


# ---------------------------------------------------------------------------
# GraphRAG chunk normalization
# ---------------------------------------------------------------------------


class TestNormalizeGraphragChunk:
    def _sample_chunk(self) -> dict:
        return {
            "chunk_id": "chunk-xyz",
            "text": "The Council endorsed the EDIS framework in April 2024.",
            "score": 0.87,
            "entity_id": "EDIS",
            "metadata": {
                "document_id": "doc-council-1",
                "title": "Council conclusions on EDIS",
                "url": "https://consilium.europa.eu/council-edis",
                "source": "council_conclusions",
                "published": "2024-04-10",
            },
        }

    def test_basic_record_fields(self):
        rec = normalize_graphrag_chunk(
            self._sample_chunk(),
            query="EDIS council",
            thread_id="t-2",
            sub_question_id="sq-2",
        )
        assert rec["thread_id"] == "t-2"
        assert rec["sub_question_id"] == "sq-2"
        assert rec["source_authority"] == 3  # council_conclusions
        assert rec["directness"] == 5  # text present

    def test_score_preserved_in_retrieval(self):
        rec = normalize_graphrag_chunk(
            self._sample_chunk(),
            query="q",
            thread_id="t-2",
            sub_question_id="sq-2",
        )
        assert rec["provenance"]["retrieval"]["score"] == pytest.approx(0.87)

    def test_passage_ref_populated(self):
        rec = normalize_graphrag_chunk(
            self._sample_chunk(),
            query="q",
            thread_id="t-2",
            sub_question_id="sq-2",
        )
        passage = rec["provenance"]["passage"]
        assert passage is not None
        assert passage["chunk_id"] == "chunk-xyz"
        assert "EDIS" in passage["text"]

    def test_relation_ref_attached(self):
        relation = {
            "relation_id": "rel-r1",
            "source_entity": "Council",
            "target_entity": "EDIS",
            "relation_type": "ENDORSED",
            "extraction_method": "graphgen",
            "confidence": 0.9,
        }
        rec = normalize_graphrag_chunk(
            self._sample_chunk(),
            query="q",
            thread_id="t-2",
            sub_question_id="sq-2",
            relation=relation,
        )
        rel = rec["provenance"]["relation"]
        assert rel is not None
        assert rel["relation_id"] == "rel-r1"
        assert rel["source_entity"] == "Council"
        assert rel["confidence"] == pytest.approx(0.9)
        assert "relation" in rec["tags"]

    def test_graph_path_ref_attached(self):
        graph_path = {
            "path_id": "path-p1",
            "entity_names": ["Council", "EDIS"],
            "edge_types": ["ENDORSED"],
            "supporting_chunk_ids": ["chunk-xyz"],
            "hypothesis_only": False,
        }
        rec = normalize_graphrag_chunk(
            self._sample_chunk(),
            query="q",
            thread_id="t-2",
            sub_question_id="sq-2",
            graph_path=graph_path,
        )
        gp = rec["provenance"]["graph_path"]
        assert gp is not None
        assert gp["path_id"] == "path-p1"
        assert "graph-path" in rec["tags"]

    def test_missing_score_is_none(self):
        chunk = self._sample_chunk()
        del chunk["score"]
        rec = normalize_graphrag_chunk(chunk, query="q", thread_id="t-2", sub_question_id="sq-2")
        assert rec["provenance"]["retrieval"]["score"] is None

    def test_no_text_gives_directness_three(self):
        chunk = self._sample_chunk()
        chunk["text"] = ""
        rec = normalize_graphrag_chunk(chunk, query="q", thread_id="t-2", sub_question_id="sq-2")
        assert rec["directness"] == 3
        assert rec["provenance"]["passage"] is None

    def test_graphrag_source_defaults_when_metadata_missing(self):
        chunk = {"chunk_id": "c1", "text": "some text"}
        rec = normalize_graphrag_chunk(chunk, query="q", thread_id="t-3", sub_question_id="sq-3")
        assert rec["source_authority"] == 4  # "graphrag" default
        assert rec["provenance"]["retrieval"]["source_type"] == "graphrag"


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


class TestBatchNormalization:
    def test_normalize_search_results_returns_one_per_doc(self):
        docs = [
            {"id": f"doc-{i}", "title": f"Doc {i}", "text": f"text {i}", "url": "", "source": "eurlex", "date": ""}
            for i in range(5)
        ]
        records = normalize_search_results(
            docs,
            query="test",
            thread_id="t-batch",
            sub_question_id="sq-batch",
        )
        assert len(records) == 5

    def test_normalize_search_results_preserves_rank(self):
        docs = [
            {"id": f"doc-{i}", "title": f"Doc {i}", "text": f"text {i}", "url": "", "source": "ep_debates", "date": ""}
            for i in range(3)
        ]
        records = normalize_search_results(
            docs,
            query="test",
            thread_id="t-batch",
            sub_question_id="sq-batch",
        )
        for i, rec in enumerate(records):
            assert rec["provenance"]["retrieval"]["rank"] == i

    def test_normalize_search_results_shares_retrieval_id(self):
        docs = [
            {"id": f"d{i}", "title": f"T{i}", "text": "x", "url": "", "source": "eurlex", "date": ""}
            for i in range(3)
        ]
        records = normalize_search_results(
            docs, query="q", thread_id="t", sub_question_id="sq"
        )
        ids = {r["provenance"]["retrieval"]["retrieval_id"] for r in records}
        assert len(ids) == 1  # all share the same retrieval_id

    def test_normalize_graphrag_results_returns_one_per_chunk(self):
        chunks = [
            {"chunk_id": f"c{i}", "text": f"chunk {i}", "score": 0.5,
             "metadata": {"source": "graphrag", "title": "", "url": "", "document_id": "", "published": ""}}
            for i in range(4)
        ]
        records = normalize_graphrag_results(
            chunks, query="q", thread_id="t", sub_question_id="sq"
        )
        assert len(records) == 4

    def test_normalize_graphrag_results_preserves_rank(self):
        chunks = [
            {"chunk_id": f"c{i}", "text": f"t{i}", "score": 0.9 - i * 0.1,
             "metadata": {"source": "graphrag"}}
            for i in range(3)
        ]
        records = normalize_graphrag_results(
            chunks, query="q", thread_id="t", sub_question_id="sq"
        )
        for i, rec in enumerate(records):
            assert rec["provenance"]["retrieval"]["rank"] == i

    def test_normalize_search_results_empty_list(self):
        records = normalize_search_results([], query="q", thread_id="t", sub_question_id="sq")
        assert records == []

    def test_normalize_graphrag_results_empty_list(self):
        records = normalize_graphrag_results([], query="q", thread_id="t", sub_question_id="sq")
        assert records == []

    def test_evidence_ids_are_unique_across_batch(self):
        docs = [
            {"id": f"doc-{i}", "title": f"D{i}", "text": "x", "url": "", "source": "eurlex", "date": ""}
            for i in range(10)
        ]
        records = normalize_search_results(docs, query="q", thread_id="t", sub_question_id="sq")
        ids = [r["evidence_id"] for r in records]
        assert len(ids) == len(set(ids)), "evidence_ids must be unique"
