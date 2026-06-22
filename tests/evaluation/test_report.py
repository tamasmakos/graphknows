"""Unit tests for evaluation report writing."""

from __future__ import annotations

from evaluation.config import EvalConfig
from evaluation.datamodels import CaseResult, RetrievedPassage, RunReport
from evaluation.reporting import summarize, write_report


def _case(score: float, category: str) -> CaseResult:
    return CaseResult(
        case_id=f"conv-26-q-{category}",
        question="Where did they go?",
        gold="Prague",
        category=category,
        dialogue_id="conv-26",
        passages=[RetrievedPassage(text="Prague", score=0.4, sources="ltm_graph")],
        generated_answer="Prague",
        score=score,
        metrics={"gold_reachable_at_20": True, "recall_hit_count": 1},
    )


def test_summarize_aggregates_scores_and_categories() -> None:
    summary = summarize([_case(1.0, "single_hop"), _case(0.0, "temporal")])
    assert summary["case_count"] == 2
    assert summary["avg_score"] == 0.5
    assert summary["avg_score_by_category"]["single_hop"] == 1.0
    assert summary["avg_score_by_category"]["temporal"] == 0.0
    assert summary["metric_summary"]["recall_hit_count"] == 1.0


def test_write_report_emits_json_csv_markdown(tmp_path) -> None:
    cases = [_case(1.0, "single_hop")]
    config = EvalConfig(output_dir=tmp_path)
    run = RunReport(run_id="eval-test", config=config, cases=cases, summary=summarize(cases))

    paths = write_report(run)

    assert paths.json_path.exists()
    assert paths.csv_path.exists()
    assert paths.markdown_path.exists()

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Mean token-F1" in markdown
    assert "single_hop" in markdown

    csv_text = paths.csv_path.read_text(encoding="utf-8")
    assert "case_id" in csv_text
    assert "Prague" in csv_text
