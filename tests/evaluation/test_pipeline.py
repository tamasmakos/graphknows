"""Smoke tests for the pipeline stages against a fake MCP client."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.config import EvalConfig
from evaluation.datamodels import EvalDocument, IngestResult, LocomoCase
from evaluation.ingestion import Ingestor
from evaluation.pipeline import _build_case_result
from evaluation.retrieval import Retriever


class FakeClient:
    """Records tool calls and returns canned responses keyed by tool name."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        return self._responses.get(name, {})


@pytest.mark.asyncio
async def test_ingestor_aggregates_and_flushes() -> None:
    client = FakeClient(
        {
            "memory_ingest": {"chunks": 3, "entities": 2, "elapsed_s": 0.1},
            "memory_flush": {"nodes": 5, "edges": 4, "elapsed_s": 0.2, "errors": []},
            "ltm_entities": {"entities": [{"name": "Alex"}, {"name": "Prague"}]},
        }
    )
    config = EvalConfig(promote_to_ltm=True)
    ingestor = Ingestor(client, config)  # type: ignore[arg-type]

    result = await ingestor.ingest("sess-1", [EvalDocument(title="s1", text="hello")])

    assert result.chunks == 3
    assert result.entities == 2
    assert result.flush_nodes == 5
    assert result.ltm_entity_count == 2
    assert [c[0] for c in client.calls] == ["memory_ingest", "memory_flush", "ltm_entities"]


@pytest.mark.asyncio
async def test_retriever_parses_passages() -> None:
    client = FakeClient(
        {"memory_query": {"hits": [{"text": "Prague", "score": 0.4, "sources": "ltm_graph"}]}}
    )
    retriever = Retriever(client, EvalConfig())  # type: ignore[arg-type]

    passages = await retriever.query("sess-1", "where?", k=5)

    assert len(passages) == 1
    assert passages[0].text == "Prague"
    assert passages[0].is_ltm_graph is True


def test_build_case_result_scores_token_f1() -> None:
    case = LocomoCase(
        case_id="conv-26-q0001",
        dialogue_id="conv-26",
        question="Where?",
        gold="Prague",
        category="single_hop",
    )
    result = _build_case_result(
        case,
        config=EvalConfig(),
        session_id="sess-1",
        passages=[],
        probe=[],
        generated_answer="Prague",
        gen_usage={"gen_total_tokens": 42},
        judge_correct=True,
        ingest=IngestResult(chunks=3),
    )
    assert result.score == 1.0
    assert result.metrics["score_type"] == "token_f1"
    assert result.metrics["judge_correct"] is True
    assert result.metrics["gen_total_tokens"] == 42
    assert result.metrics["ingested_chunks"] == 3
