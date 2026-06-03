"""Unit tests for evaluation report writing."""

from __future__ import annotations

from evaluation.core.reporting.metrics import ToolObservation
from evaluation.core.reporting import write_benchmark_report, write_report


def test_write_report_includes_tool_rows(tmp_path) -> None:
    paths = write_report(
        [ToolObservation("case-1", "memory_stats", True, 5.0, {"stm": {}, "ltm": {}})],
        output_dir=tmp_path,
        run_id="eval-test",
    )

    assert paths.json_path.exists()
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "memory_stats" in markdown
    assert "Tool Summary" in markdown


def test_write_benchmark_report_aggregates_metrics(tmp_path) -> None:
    paths = write_benchmark_report(
        benchmark="hotpotqa",
        run_id="eval-benchmark",
        output_dir=tmp_path,
        cases=[
            {
                "case_id": "row-1",
                "question": "Who wrote the book?",
                "gold": "Ada",
                "memory_context": [],
                "metrics": {"answer_support_hit": True, "supporting_title_recall_at_k": 0.5},
                "score": None,
                "metadata": {"stm_snapshot": {"total_stm_nodes": 12}},
            }
        ],
    )

    assert paths.json_path.exists()
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Aggregated Metrics" in markdown
    assert "supporting_title_recall_at_k" in markdown
