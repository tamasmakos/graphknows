"""The panel reports the local and LLM extraction paths as separate columns (FR-043)."""

from __future__ import annotations

from evaluation.common.reporting import EXTRACTOR_COLUMNS, PanelRow, render_panel


def _row(name: str, extractor: str, accuracy: float, recall: float) -> PanelRow:
    return PanelRow(
        row=name,
        feeding_mode="turn_by_turn",
        extractor=extractor,
        accuracy=accuracy,
        evidence_recall=recall,
    )


class TestRenderPanel:
    def test_header_carries_a_column_per_extraction_path(self) -> None:
        header = render_panel([_row("locomo", "local", 0.5, 0.4)]).splitlines()[2]
        for extractor in EXTRACTOR_COLUMNS:
            assert f"{extractor} accuracy" in header
            assert f"{extractor} evidence_recall" in header

    def test_both_paths_of_a_row_share_one_line(self) -> None:
        table = render_panel([_row("locomo", "local", 0.5, 0.4), _row("locomo", "llm", 0.7, 0.6)])
        lines = [line for line in table.splitlines() if line.startswith("| `locomo`")]
        assert len(lines) == 1
        assert lines[0] == "| `locomo` | 0.500 | 0.400 | 0.700 | 0.600 |"

    def test_a_path_that_did_not_run_is_named_not_omitted(self) -> None:
        rows = [
            _row("repo", "local", 0.3, 0.2),
            PanelRow(row="repo", feeding_mode="", extractor="llm", not_run="no credential"),
        ]
        line = next(text for text in render_panel(rows).splitlines() if text.startswith("| `repo`"))
        assert "not run (no credential)" in line

    def test_a_missing_path_leaves_an_empty_cell(self) -> None:
        line = render_panel([_row("beam", "local", 0.1, 0.2)]).splitlines()[-1]
        assert line == "| `beam` | 0.100 | 0.200 | — | — |"
