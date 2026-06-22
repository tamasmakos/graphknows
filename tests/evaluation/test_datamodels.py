"""Unit tests for the evaluation datamodels."""

from __future__ import annotations

from evaluation.config import EvalConfig
from evaluation.datamodels import CaseResult, RetrievedPassage, RunReport


def test_retrieved_passage_detects_ltm_graph_source() -> None:
    assert RetrievedPassage(text="x", sources="ltm_graph,semantic").is_ltm_graph is True
    assert RetrievedPassage(text="x", sources="stm_text").is_ltm_graph is False


def test_case_result_to_record_includes_memory_context() -> None:
    case = CaseResult(
        case_id="conv-26-q0001",
        question="Where?",
        gold="Prague",
        category="single_hop",
        dialogue_id="conv-26",
        passages=[RetrievedPassage(text="Prague", score=0.5)],
        generated_answer="Prague",
        score=1.0,
    )
    record = case.to_record()
    assert record["case_id"] == "conv-26-q0001"
    assert record["score"] == 1.0
    assert record["memory_context"][0]["text"] == "Prague"


def test_eval_config_is_frozen() -> None:
    config = EvalConfig()
    try:
        config.top_k = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("EvalConfig should be frozen")


def test_run_report_round_trips() -> None:
    report = RunReport(run_id="eval-1", config=EvalConfig(), cases=[], summary={"case_count": 0})
    dumped = report.model_dump()
    assert dumped["run_id"] == "eval-1"
    assert dumped["config"]["top_k"] == 5
