"""Tests for deterministic research-memory evaluation helpers."""

from __future__ import annotations

from evaluation.core.research.harness import (
    ResearchCase,
    ResearchMemoryOptions,
    _score_case,
    run_research_memory,
)


def test_score_case_reports_headline_scores() -> None:
    case = ResearchCase(
        case_id="c1",
        question="Who supports the AI Act?",
        gold="Council",
        documents=[],
        expected_answer_terms=("Council",),
        expected_evidence_terms=("AI Act",),
        expected_entity_terms=("Council",),
        metadata={},
    )
    retrievals = {
        "memory_recall_ltm": {
            "hits": [{"text": "The Council supports the AI Act.", "entities": ["Council"]}]
        },
        "recall_by_document": {"results": []},
        "recall_by_topic": {"results": [{"text": "AI Act policy", "entities": ["Council"]}]},
        "ltm_query": {"results": []},
        "ltm_topics": {"topics": []},
    }

    metrics = _score_case(case, retrievals)

    assert metrics["answer_support_score"] == 1.0
    assert metrics["evidence_recall_score"] == 1.0
    assert metrics["topic_entity_score"] == 1.0
    assert metrics["research_overall_score"] == 1.0


async def test_run_research_memory_uses_fixed_operation_sequence(monkeypatch) -> None:
    case = ResearchCase(
        case_id="c1",
        question="Who supports the AI Act?",
        gold="Council",
        documents=[{"id": "d1", "title": "Doc", "source": "test", "text": "Council supports it."}],
        expected_answer_terms=("Council",),
        expected_evidence_terms=("Council",),
        expected_entity_terms=("Council",),
        metadata={"dataset": "synthetic"},
    )

    async def fake_build_cases(options):
        return [case]

    import evaluation.core.research as mod

    monkeypatch.setattr(mod, "build_research_cases", fake_build_cases)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            if name == "corpus_ingest":
                return {
                    "documents_processed": 1,
                    "entities_extracted": 1,
                    "chunks_written": 1,
                    "nodes_written": 1,
                    "edges_written": 1,
                    "errors": [],
                }
            if name == "memory_recall":
                return {"hits": [{"text": "Council supports the AI Act.", "entities": ["Council"]}]}
            if name == "recall_by_topic":
                return {"results": [{"text": "AI Act", "entities": ["Council"]}]}
            if name == "ltm_topics":
                return {"topics": [{"summary": "AI Act", "entity_names": ["Council"]}]}
            if name == "ltm_query":
                return {"results": [{"summary": "AI Act", "entity_names": ["Council"]}]}
            if name == "recall_by_document":
                return {"results": [{"text": "Council supports the AI Act."}]}
            return {"errors": []}

    client = FakeClient()
    options = ResearchMemoryOptions(
        suite="hotpotqa",
        run_id="run",
        top_k=5,
    )

    results = await run_research_memory(client, options)  # type: ignore[arg-type]

    assert [name for name, _args in client.calls] == [
        "memory_purge",
        "corpus_ingest",
        "memory_stats",
        "memory_recall",
        "recall_by_document",
        "recall_by_topic",
        "ltm_query",
        "ltm_topics",
        "memory_purge",
    ]
    old_schema_arg = "taxonomy" + "_file"
    assert old_schema_arg not in client.calls[1][1]
    assert client.calls[3][1]["scope"] == "ltm"
    assert results[0].metrics["answer_support_score"] == 1.0
