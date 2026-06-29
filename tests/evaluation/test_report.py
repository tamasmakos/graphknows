"""Unit tests for evaluation report writing."""

from __future__ import annotations

from evaluation.config import EvalConfig
from evaluation.datamodels import CaseResult, RetrievedPassage, RunReport
from evaluation.reporting import compare_modes, summarize, write_report


def _case(score: float, category: str, semantic_f1: float | None = None) -> CaseResult:
    metrics: dict = {"gold_reachable_at_20": True, "recall_hit_count": 1}
    # If semantic_f1 is provided explicitly, store it; otherwise use score as proxy.
    if semantic_f1 is not None:
        metrics["semantic_f1"] = semantic_f1
        metrics["token_f1"] = score
    return CaseResult(
        case_id=f"conv-26-q-{category}",
        question="Where did they go?",
        gold="Prague",
        category=category,
        dialogue_id="conv-26",
        passages=[RetrievedPassage(text="Prague", score=0.4, sources="ltm_graph")],
        generated_answer="Prague",
        score=semantic_f1 if semantic_f1 is not None else score,
        metrics=metrics,
    )


def test_summarize_headline_is_semantic_f1() -> None:
    summary = summarize([
        _case(1.0, "single_hop", semantic_f1=0.9),
        _case(0.0, "temporal", semantic_f1=0.5),
    ])
    assert summary["case_count"] == 2
    assert abs(summary["mean_semantic_f1"] - 0.7) < 1e-6
    assert abs(summary["mean_token_f1"] - 0.5) < 1e-6  # avg of 1.0, 0.0
    # Per-category breakdown.
    assert abs(summary["semantic_f1_by_category"]["single_hop"] - 0.9) < 1e-6
    assert abs(summary["semantic_f1_by_category"]["temporal"] - 0.5) < 1e-6
    # Legacy alias.
    assert summary["avg_score"] == summary["mean_semantic_f1"]
    # judge_accuracy must not be present.
    assert "judge_accuracy" not in summary


def test_summarize_aggregates_categories() -> None:
    summary = summarize([_case(1.0, "single_hop"), _case(0.0, "temporal")])
    assert summary["case_count"] == 2
    assert "single_hop" in summary["semantic_f1_by_category"]
    assert "temporal" in summary["semantic_f1_by_category"]
    assert summary["metric_summary"]["recall_hit_count"] == 1.0


def test_write_report_emits_json_csv_markdown(tmp_path) -> None:
    cases = [_case(1.0, "single_hop", semantic_f1=0.9)]
    config = EvalConfig(output_dir=tmp_path)
    run = RunReport(run_id="eval-test", config=config, cases=cases, summary=summarize(cases))

    paths = write_report(run)

    assert paths.json_path.exists()
    assert paths.csv_path.exists()
    assert paths.markdown_path.exists()

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "SemanticF1" in markdown
    assert "single_hop" in markdown

    csv_text = paths.csv_path.read_text(encoding="utf-8")
    assert "case_id" in csv_text
    assert "semantic_f1" in csv_text
    assert "token_f1" in csv_text
    assert "Prague" in csv_text
    # judge_correct column must not be present.
    assert "judge_correct" not in csv_text


def test_compare_modes_table_shape() -> None:
    free_cases = [
        _case(0.5, "single_hop", semantic_f1=0.6),
        _case(0.3, "temporal", semantic_f1=0.4),
    ]
    asst_cases = [
        _case(0.7, "single_hop", semantic_f1=0.8),
        _case(0.5, "temporal", semantic_f1=0.6),
    ]
    table = compare_modes(free_cases, asst_cases)
    assert "single_hop" in table
    assert "temporal" in table
    assert "Overall" in table
    assert "llm_free" in table
    assert "llm_assisted" in table
