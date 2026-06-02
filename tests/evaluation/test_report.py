"""Unit tests for evaluation report writing."""

from __future__ import annotations

from evaluation.metrics import ToolObservation
from evaluation.report import write_report


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

