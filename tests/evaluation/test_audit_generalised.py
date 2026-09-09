"""Fact precision is a per-run metric on any corpus, dialogue or not (FR-040).

The audit judges REL edges against their evidence chunks, which says nothing
about where the chunks came from. What was dialogue-shaped was the reporting: the
number lived in a hand-run LoCoMo report. Here it is pooled per panel row, and a
row with nothing judged still reports the metric with its reason.
"""

from __future__ import annotations

from evaluation.audit.core import (
    JOINT_AXIS,
    aggregate,
    build_report,
    fact_precision,
    render_markdown,
)


def _row(tier: str, verdicts: tuple[str, str, str, str]) -> dict[str, str]:
    span, predicate, direction, polarity = verdicts
    return {
        "rid": f"#1:{tier}",
        "tier": tier,
        "evidence_chunk_ids": "chunk-1",
        "judge_span": span,
        "judge_predicate": predicate,
        "judge_direction": direction,
        "judge_polarity_modality": polarity,
    }


_ALL_RIGHT = ("1", "1", "1", "1")
_WRONG_DIRECTION = ("1", "1", "0", "1")


class TestFactPrecision:
    def test_pools_the_joint_axis_over_every_tier(self) -> None:
        tiers = aggregate(
            [
                _row("frame", _ALL_RIGHT),
                _row("ontology", _ALL_RIGHT),
                _row("lemma", _WRONG_DIRECTION),
            ]
        )
        result = fact_precision("repo", tiers)
        assert (result.judged, result.correct) == (3, 2)
        assert result.precision == 2 / 3
        assert result.not_judged is None

    def test_keeps_the_per_tier_breakdown(self) -> None:
        tiers = aggregate([_row("frame", _ALL_RIGHT), _row("lemma", _WRONG_DIRECTION)])
        by_tier = fact_precision("longmem", tiers).by_tier
        assert by_tier["frame"] == 1.0
        assert by_tier["lemma"] == 0.0
        assert by_tier["wordnet"] is None

    def test_any_row_is_named_not_just_the_dialogue_one(self) -> None:
        tiers = aggregate([_row("frame", _ALL_RIGHT)])
        assert fact_precision("beam", tiers).row == "beam"

    def test_a_row_without_a_sheet_still_reports_the_metric(self) -> None:
        result = fact_precision("repo", [])
        assert result.precision is None
        assert result.judged == 0
        assert "repo" in str(result.not_judged)

    def test_an_unjudged_sheet_is_distinguished_from_a_missing_one(self) -> None:
        result = fact_precision("repo", aggregate([_row("frame", ("", "", "", ""))]))
        assert result.precision is None
        assert "no judged rows" in str(result.not_judged)

    def test_record_matches_the_report_contract(self) -> None:
        record = fact_precision("repo", aggregate([_row("frame", _ALL_RIGHT)])).to_record()
        assert record["row"] == "repo"
        assert record["precision"] == 1.0
        assert record["judged"] == 1
        assert record["not_judged"] is None
        assert record["by_tier"]["frame"] == 1.0


class TestReport:
    def test_every_report_carries_a_fact_precision_block(self) -> None:
        report = build_report(aggregate([_row("frame", _ALL_RIGHT)]), {"row": "repo"})
        assert report["fact_precision"]["row"] == "repo"
        assert report["fact_precision"]["precision"] == 1.0

    def test_a_report_without_a_row_says_unknown_rather_than_omitting_it(self) -> None:
        report = build_report(aggregate([]), {})
        assert report["fact_precision"]["row"] == "unknown"
        assert report["fact_precision"]["precision"] is None

    def test_markdown_states_the_number_and_the_pooled_sample(self) -> None:
        report = build_report(aggregate([_row("frame", _ALL_RIGHT)]), {"row": "repo"})
        assert "Fact precision: `repo` 1.00" in render_markdown(report)

    def test_markdown_states_the_reason_when_nothing_was_judged(self) -> None:
        rendered = render_markdown(build_report([], {"row": "repo"}))
        assert "Fact precision: — (no audit sheet for row 'repo')" in rendered


def test_the_joint_axis_is_what_fact_precision_reports() -> None:
    """A triplet is only a fact if all four axes hold, so the metric is the conjunction."""
    tiers = aggregate([_row("frame", _WRONG_DIRECTION)])
    assert tiers[0].axes[JOINT_AXIS].precision == 0.0
    assert fact_precision("locomo", tiers).precision == 0.0
