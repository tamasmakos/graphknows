"""Pipeline stages against a fake MCP client, plus reporting aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import (
    CaseResult,
    EvalCase,
    EvalDocument,
    IngestResult,
    RetrievedPassage,
    RunReport,
)
from evaluation.common.ingestion import Ingestor
from evaluation.common.pipeline import build_case_result, unit_namespace
from evaluation.common.reporting import compare_modes, summarize, write_report
from evaluation.common.retrieval import Retriever


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
            "memory_ingest": {"chunks": 3, "entities": 2, "elapsed_s": 0.5},
            "memory_flush": {"nodes": 10, "edges": 4, "chunks": 3, "elapsed_s": 1.0},
        }
    )
    ingestor = Ingestor(client, BaseEvalConfig())  # type: ignore[arg-type]
    result = await ingestor.ingest(
        "s1",
        [EvalDocument(title="d1", text="hello"), EvalDocument(title="d2", text="world")],
        namespace="eval_locomo_llm_free_conv_26",
    )
    assert result.chunks == 6 and result.entities == 4
    assert result.flush_nodes == 10
    assert [name for name, _ in client.calls] == [
        "memory_ingest",
        "memory_ingest",
        "memory_flush",
    ]
    # Every call is routed to the unit's own namespace.
    assert all(args.get("namespace") == "eval_locomo_llm_free_conv_26" for _, args in client.calls)


@pytest.mark.asyncio
async def test_turn_fed_unit_sends_messages_and_still_flushes() -> None:
    """A turn-fed unit buffers TURN rows; its graph only exists after flush.

    Sending `text` here would run eager document extraction and put the harness
    straight back on session-sized chunks — the thing turn-feeding exists to
    avoid. `chunks` is legitimately 0 at ingest time.
    """
    client = FakeClient(
        {
            "memory_ingest": {"chunks": 0, "entities": 0, "turns": 2, "elapsed_s": 0.1},
            "memory_flush": {"nodes": 7, "edges": 3, "chunks": 4, "elapsed_s": 1.0},
        }
    )
    ingestor = Ingestor(client, BaseEvalConfig())  # type: ignore[arg-type]
    doc = EvalDocument(
        title="d1",
        text="Gina: I lost my job.\nJon: Sorry.",
        turns=[
            {"role": "user", "content": "I lost my job.", "name": "Gina", "timestamp": "1 May"},
            {"role": "user", "content": "Sorry.", "name": "Jon", "timestamp": "1 May"},
        ],
    )

    result = await ingestor.ingest("s1", [doc], namespace="ns")

    ingest_args = next(args for name, args in client.calls if name == "memory_ingest")
    assert "text" not in ingest_args
    assert [m["content"] for m in ingest_args["messages"]] == ["I lost my job.", "Sorry."]
    assert [m["name"] for m in ingest_args["messages"]] == ["Gina", "Jon"]
    # Buffered, not extracted — the real counts arrive from flush.
    assert result.turns == 2
    assert result.chunks == 0
    assert result.flush_chunks == 4
    assert result.ingest_error_count == 0
    assert "memory_flush" in [name for name, _ in client.calls]


def test_unit_namespace_is_per_conversation() -> None:
    config = BaseEvalConfig().model_copy(update={"namespace": "eval_locomo_llm_free"})
    assert unit_namespace(config, "conv-26") == "eval_locomo_llm_free_conv-26"
    assert unit_namespace(config, "conv-30") == "eval_locomo_llm_free_conv-30"


@pytest.mark.asyncio
async def test_retriever_parses_hits() -> None:
    client = FakeClient(
        {"memory_query": {"hits": [{"text": "t", "score": 0.9, "sources": "entity"}]}}
    )
    retriever = Retriever(client, BaseEvalConfig())  # type: ignore[arg-type]
    passages, facts = await retriever.query(
        "s1", "q", k=5, namespace="eval_locomo_llm_free_conv_26"
    )
    assert passages[0].text == "t" and passages[0].is_graph_hit
    assert facts == []
    assert client.calls[0][1]["top_k"] == 5
    assert client.calls[0][1]["namespace"] == "eval_locomo_llm_free_conv_26"


@pytest.mark.asyncio
async def test_gen_context_k_100_probes_and_slices_100() -> None:
    """End-to-end depth plumbing: gen_context_k=100 → probe_k>=100 → the retriever
    requests >=100 passages and the generator context is sliced to exactly 100."""
    config = BaseEvalConfig(gen_context_k=100)
    assert config.probe_k >= 100  # validator coverage
    hits = [{"text": f"p{i}", "score": 1.0 - i / 200} for i in range(120)]
    client = FakeClient({"memory_query": {"hits": hits}})
    retriever = Retriever(client, config)  # type: ignore[arg-type]
    probe, _facts = await retriever.query("s1", "q", k=config.probe_k, namespace="ns")
    # The MCP tool caps at 100; here the fake returns everything, so assert the
    # eval REQUESTED at least 100 (not silently capped at 25/50 on our side).
    assert client.calls[0][1]["top_k"] >= 100
    context_passages = probe[: config.gen_context_k]
    assert len(context_passages) == 100


def _result(score: float, category: str = "temporal", **metrics: Any) -> CaseResult:
    return CaseResult(
        case_id=f"c-{score}-{category}",
        question="q",
        gold="g",
        category=category,
        group_id="g1",
        score=score,
        generated_answer="a",
        metrics={"judge_score": score, **metrics},
    )


def test_build_case_result_diagnostics() -> None:
    case = EvalCase(case_id="c1", group_id="g1", question="q", gold="Lake Tahoe")
    passages = [RetrievedPassage(text="We kayaked at Lake Tahoe", score=0.8)]
    result = build_case_result(
        case,
        config=BaseEvalConfig(),
        session_id="s1",
        passages=passages,
        probe=passages,
        context_passages=passages,
        generated_answer="Lake Tahoe",
        score=1.0,
        judge_metrics={"judge_label": "CORRECT"},
        lexical={},
        gen_usage={"gen_total_tokens": 10},
        ingest=IngestResult(chunks=5),
    )
    assert result.score == 1.0
    assert result.metrics["gold_reachable_at_probe"] is True
    assert result.metrics["gold_context_coverage"] == 1.0
    assert result.metrics["ingested_chunks"] == 5
    assert result.metadata["session_id"] == "s1"


def test_summarize_accuracy_and_categories() -> None:
    cases = [_result(1.0), _result(0.0), _result(1.0, category="single_hop")]
    summary = summarize(cases)
    assert summary["case_count"] == 3
    assert summary["accuracy"] == round(2 / 3, 4)
    assert summary["accuracy_by_category"]["temporal"] == 0.5
    assert summary["accuracy_by_category"]["single_hop"] == 1.0
    assert summary["category_counts"]["temporal"] == 2


def test_write_report_produces_three_files(tmp_path: Path) -> None:
    run = RunReport(
        run_id="r1",
        benchmark="locomo",
        config={"top_k": 5},
        cases=[_result(1.0, locomo_f1=0.8)],
        summary=summarize([_result(1.0, locomo_f1=0.8)]),
    )
    paths = write_report(run, tmp_path)
    assert paths.json_path.exists() and paths.csv_path.exists() and paths.markdown_path.exists()
    assert "locomo_f1" in paths.csv_path.read_text(encoding="utf-8")
    assert "Accuracy" in paths.markdown_path.read_text(encoding="utf-8")


def test_compare_modes_table() -> None:
    table = compare_modes([_result(1.0)], [_result(0.0)])
    assert "| `temporal` | 1.000 | 0.000 | -1.000 |" in table
    assert "**Overall**" in table
