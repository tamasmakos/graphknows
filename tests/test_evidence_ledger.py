"""Tests for evidence ledger normalization and extension helpers."""
from __future__ import annotations

from services.agent.graph.ledger import (
    extend_ledger,
    normalize_graphrag_results,
    normalize_search_results,
)


def test_normalize_search_results_populates_evidence_record_fields() -> None:
    raw = {
        "documents": [
            {
                "id": "doc-1",
                "title": "EP plenary debate on energy security",
                "text": "Parliament debated energy resilience in spring session.",
                "url": "https://www.europarl.europa.eu/debate/doc-1",
                "source": "ep_debates",
                "date": "2025-01-15",
            }
        ],
        "count": 1,
        "total_fetched": 1,
    }

    records = normalize_search_results(
        raw,
        retrieval_id="ret-search-1",
        query="energy resilience plenary debate",
        thread_id="thread-1",
        sub_question_id="sq-1",
    )

    assert len(records) == 1
    record = records[0]
    assert isinstance(record["evidence_id"], str)
    assert record["evidence_id"]
    assert record["thread_id"] == "thread-1"
    assert record["sub_question_id"] == "sq-1"
    assert record["summary"] == "Parliament debated energy resilience in spring session."
    assert record["source_authority"] == 5
    assert record["independence"] == 3
    assert record["directness"] == 3
    assert record["recency"] == 3
    assert record["claim_ids"] == []
    assert record["tags"] == []
    assert record["provenance"]["retrieval"]["retrieval_id"] == "ret-search-1"
    assert record["provenance"]["retrieval"]["query"] == "energy resilience plenary debate"
    assert record["provenance"]["retrieval"]["tool_name"] == "search_eu"
    assert record["provenance"]["retrieval"]["rank"] == 0
    assert record["provenance"]["retrieval"]["score"] is None
    assert record["provenance"]["retrieval"]["source_type"] == "ep_debates"
    assert record["provenance"]["document"]["document_id"] == "doc-1"
    assert record["provenance"]["document"]["title"] == "EP plenary debate on energy security"
    assert record["provenance"]["document"]["publisher"] == "ep_debates"
    assert record["provenance"]["document"]["published_at"] == "2025-01-15"
    assert record["provenance"]["passage"] is None
    assert record["provenance"]["relation"] is None
    assert record["provenance"]["graph_path"] is None
    assert record["provenance"]["retrieval"]["retrieved_at"]


def test_normalize_graphrag_results_populates_passage_and_graph_path() -> None:
    raw = {
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "text": "The committee report links investment targets to net-zero milestones.",
                "score": 0.91,
                "metadata": {
                    "document_id": "doc-a",
                    "title": "Committee report",
                    "url": "https://example.org/report",
                    "source": "eurlex",
                    "published_at": "2024-11-10",
                },
            }
        ],
        "paths": [
            {
                "path_id": "path-1",
                "entity_names": ["Investment Plan", "Net Zero"],
                "edge_types": ["SUPPORTS"],
                "supporting_chunk_ids": ["chunk-1"],
                "hypothesis_only": False,
            }
        ],
    }

    records = normalize_graphrag_results(
        raw,
        retrieval_id="ret-gr-1",
        query="investment target support",
        thread_id="thread-2",
        sub_question_id="sq-2",
    )

    assert len(records) == 1
    record = records[0]
    assert record["source_authority"] == 5
    assert record["summary"] == "The committee report links investment targets to net-zero milestones."
    assert record["provenance"]["retrieval"]["score"] == 0.91
    assert record["provenance"]["retrieval"]["tool_name"] == "retrieve"
    passage = record["provenance"]["passage"]
    assert passage is not None
    assert passage["chunk_id"] == "chunk-1"
    assert passage["start_char"] == 0
    assert passage["end_char"] == len(
        "The committee report links investment targets to net-zero milestones."
    )
    graph_path = record["provenance"]["graph_path"]
    assert graph_path is not None
    assert graph_path["path_id"] == "path-1"
    assert graph_path["entity_names"] == ["Investment Plan", "Net Zero"]
    assert graph_path["supporting_chunk_ids"] == ["chunk-1"]
    assert record["provenance"]["relation"] is None


def test_extend_ledger_appends_without_mutating_original_state() -> None:
    original_state = {"session_id": "s-1", "evidence_ledger": []}
    new_records = normalize_search_results(
        {
            "documents": [
                {
                    "id": "doc-2",
                    "title": "Title only",
                    "text": "",
                    "url": "https://example.org/doc-2",
                    "source": "tavily",
                    "date": "",
                }
            ]
        },
        retrieval_id="ret-search-2",
        query="title query",
        thread_id="thread-3",
        sub_question_id="sq-3",
    )

    updated_state = extend_ledger(original_state, new_records)

    assert original_state["evidence_ledger"] == []
    assert len(updated_state["evidence_ledger"]) == 1
    assert updated_state is not original_state


def test_normalizers_return_empty_list_for_empty_or_missing_results() -> None:
    assert normalize_search_results(
        {},
        retrieval_id="ret-empty-1",
        query="q",
        thread_id="thread-e",
        sub_question_id="sq-e",
    ) == []
    assert normalize_search_results(
        {"documents": []},
        retrieval_id="ret-empty-2",
        query="q",
        thread_id="thread-e",
        sub_question_id="sq-e",
    ) == []
    assert normalize_graphrag_results(
        {},
        retrieval_id="ret-empty-3",
        query="q",
        thread_id="thread-e",
        sub_question_id="sq-e",
    ) == []
    assert normalize_graphrag_results(
        {"chunks": []},
        retrieval_id="ret-empty-4",
        query="q",
        thread_id="thread-e",
        sub_question_id="sq-e",
    ) == []
