"""Stratified triplet audit: tier classification, allocation, Wilson, aggregation.

All pure — no database, no filesystem. The audit's whole value is that its
numbers are honest, so the cases that matter here are the dishonest ones it must
refuse to produce: a tier that does not exist, a cell nobody judged, an interval
at n=0.
"""

from __future__ import annotations

import pytest

from evaluation.audit.cli import _instance_chunks
from evaluation.audit.core import (
    AXES,
    JOINT_AXIS,
    JUDGEMENT_COLUMNS,
    SHEET_COLUMNS,
    TIERS,
    aggregate,
    allocate,
    build_report,
    caveats,
    compare,
    parse_verdict,
    render_markdown,
    stratified_sample,
    tier_census,
    tier_of,
    wilson_interval,
)


def _edge(rid: str, canonical_predicate: str) -> dict[str, object]:
    return {"rid": rid, "canonical_predicate": canonical_predicate}


def _population(frame: int = 0, ontology: int = 0, lemma: int = 0) -> list[dict[str, object]]:
    edges = [_edge(f"#1:{i}", f"fn:Frame_{i}:A->B") for i in range(frame)]
    edges += [_edge(f"#2:{i}", "uses") for i in range(ontology)]
    edges += [_edge(f"#3:{i}", f"lemma:verb_{i}") for i in range(lemma)]
    return edges


class TestTierOf:
    def test_prefixes(self) -> None:
        assert tier_of("fn:Difficulty:Activity->Experiencer") == "frame"
        assert tier_of("wn30:00001740-n") == "wordnet"
        assert tier_of("lemma:plans to") == "lemma"

    def test_unprefixed_is_an_ontology_property(self) -> None:
        assert tier_of("uses") == "ontology"
        assert tier_of("http://example.org/ont#worksFor") == "ontology"

    def test_missing_predicate_is_not_silently_an_ontology_property(self) -> None:
        """An empty canonical_predicate is a defect, not a fifth ontology term."""
        assert tier_of(None) == "unclassified"
        assert tier_of("   ") == "unclassified"

    def test_census_keeps_empty_tiers(self) -> None:
        census = tier_census(_population(frame=3, lemma=1))
        assert census == {"frame": 3, "wordnet": 0, "ontology": 0, "lemma": 1, "unclassified": 0}
        assert set(census) == set(TIERS)


class TestAllocate:
    def test_even_split(self) -> None:
        assert allocate({"a": 100, "b": 100, "c": 100}, 150) == {"a": 50, "b": 50, "c": 50}

    def test_small_tier_hands_its_remainder_back(self) -> None:
        quota = allocate({"frame": 2472, "lemma": 166, "ontology": 99, "wordnet": 0}, 600)
        assert quota["wordnet"] == 0
        assert quota["ontology"] == 99
        assert quota["lemma"] == 166
        assert sum(quota.values()) == 600

    def test_empty_tier_never_crashes(self) -> None:
        assert allocate({"frame": 10, "wordnet": 0}, 4) == {"frame": 4, "wordnet": 0}

    def test_asks_for_more_than_exists(self) -> None:
        quota = allocate({"frame": 3, "lemma": 2}, 100)
        assert quota == {"frame": 3, "lemma": 2}

    def test_zero_and_empty(self) -> None:
        assert allocate({"frame": 10}, 0) == {"frame": 0}
        assert allocate({}, 10) == {}


class TestStratifiedSample:
    def test_deterministic_given_a_seed(self) -> None:
        edges = _population(frame=200, ontology=40, lemma=60)
        first, census, quota = stratified_sample(edges, 60, seed=7)
        second, _, _ = stratified_sample(edges, 60, seed=7)
        assert [row["rid"] for row in first] == [row["rid"] for row in second]
        assert census["frame"] == 200
        assert sum(quota.values()) == 60

    def test_different_seed_draws_differently(self) -> None:
        edges = _population(frame=200, ontology=40, lemma=60)
        first, _, _ = stratified_sample(edges, 60, seed=1)
        second, _, _ = stratified_sample(edges, 60, seed=2)
        assert [row["rid"] for row in first] != [row["rid"] for row in second]

    def test_input_order_does_not_change_the_draw(self) -> None:
        """The pool is sorted before sampling, so a reordered query result is the same sheet."""
        edges = _population(frame=50, lemma=20)
        forward, _, _ = stratified_sample(edges, 20, seed=3)
        backward, _, _ = stratified_sample(list(reversed(edges)), 20, seed=3)
        assert [row["rid"] for row in forward] == [row["rid"] for row in backward]

    def test_rows_carry_their_tier(self) -> None:
        rows, _, _ = stratified_sample(_population(frame=10, lemma=10), 8, seed=0)
        assert {row["tier"] for row in rows} == {"frame", "lemma"}

    def test_empty_tier_is_reported_not_dropped(self) -> None:
        """wn30: does not exist yet — the audit must say zero, not omit the tier."""
        _, census, quota = stratified_sample(_population(frame=10), 5, seed=0)
        assert census["wordnet"] == 0
        assert quota["wordnet"] == 0

    def test_no_edges(self) -> None:
        rows, census, quota = stratified_sample([], 10, seed=0)
        assert rows == []
        assert sum(census.values()) == sum(quota.values()) == 0


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self) -> None:
        low, high = wilson_interval(30, 37)
        assert low < 30 / 37 < high

    def test_stays_inside_the_unit_interval_at_p_equals_one(self) -> None:
        low, high = wilson_interval(37, 37)
        assert 0.0 < low < 1.0
        assert high == 1.0

    def test_p_equals_zero_has_a_non_degenerate_upper_bound(self) -> None:
        low, high = wilson_interval(0, 37)
        assert low == 0.0
        assert 0.0 < high < 0.2

    def test_no_observations_means_no_knowledge(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_width_shrinks_with_n(self) -> None:
        def width(n: int) -> float:
            low, high = wilson_interval(n // 2, n)
            return high - low

        assert width(37) > width(150) > width(1000)

    def test_thin_sample_is_about_fifteen_points_each_side(self) -> None:
        """The number the report has to be honest about: n=37 at p=0.5."""
        low, high = wilson_interval(18, 37)
        assert 0.28 < (high - low) < 0.35


class TestParseVerdict:
    @pytest.mark.parametrize("cell", ["1", "y", "Yes", "TRUE", " t "])
    def test_true_spellings(self, cell: str) -> None:
        assert parse_verdict(cell) is True

    @pytest.mark.parametrize("cell", ["0", "n", "No", "FALSE", " f "])
    def test_false_spellings(self, cell: str) -> None:
        assert parse_verdict(cell) is False

    def test_blank_is_unjudged(self) -> None:
        assert parse_verdict("") is None
        assert parse_verdict("   ") is None
        assert parse_verdict(None) is None

    def test_garbage_raises_rather_than_scoring_as_wrong(self) -> None:
        with pytest.raises(ValueError, match="unreadable judgement"):
            parse_verdict("maybe")


class TestInstanceChunks:
    """The FRAME_EVOKED_IN hop. A reified REL is an entity-adjacency projection and
    carries no evidence of its own; skipping this join reads as "2579 of 2735 edges
    have no evidence" when nearly all of them do."""

    def test_primary_chunk_leads(self) -> None:
        record = {"chunk_id": "aaa", "evoked_chunks": ["ccc", "aaa", "bbb"]}
        assert _instance_chunks(record) == ["aaa", "bbb", "ccc"]

    def test_storage_order_does_not_leak_into_the_sheet(self) -> None:
        """ArcadeDB returns evoked chunks in storage order; a re-run must not reshuffle."""
        forward = {"chunk_id": "aaa", "evoked_chunks": ["bbb", "ccc"]}
        backward = {"chunk_id": "aaa", "evoked_chunks": ["ccc", "bbb"]}
        assert _instance_chunks(forward) == _instance_chunks(backward)

    def test_primary_is_not_duplicated(self) -> None:
        assert _instance_chunks({"chunk_id": "aaa", "evoked_chunks": ["aaa"]}) == ["aaa"]

    def test_no_evoked_edges_falls_back_to_the_property(self) -> None:
        assert _instance_chunks({"chunk_id": "aaa", "evoked_chunks": []}) == ["aaa"]

    def test_nothing_at_all(self) -> None:
        assert _instance_chunks({}) == []


def _row(tier: str, verdicts: str, verifier_score: str = "") -> dict[str, str]:
    """A judged sheet row; *verdicts* is one character per axis in AXES order."""
    row = {
        "rid": f"#9:{verdicts}",
        "tier": tier,
        "verifier_score": verifier_score,
        "evidence_chunk_ids": "abc123",
    }
    row.update(dict(zip(JUDGEMENT_COLUMNS, verdicts, strict=True)))
    return row


class TestAggregate:
    def test_per_axis_precision(self) -> None:
        rows = [_row("frame", "1111"), _row("frame", "1101"), _row("frame", "1111")]
        by_tier = {tier.tier: tier for tier in aggregate(rows)}
        frame = by_tier["frame"]
        assert frame.sampled == 3
        assert frame.axes["span"].precision == 1.0
        assert frame.axes["direction"].correct == 2
        assert frame.axes["direction"].precision == pytest.approx(2 / 3)

    def test_joint_axis_is_the_conjunction(self) -> None:
        rows = [_row("frame", "1111"), _row("frame", "1011")]
        frame = {tier.tier: tier for tier in aggregate(rows)}["frame"]
        assert frame.axes[JOINT_AXIS].judged == 2
        assert frame.axes[JOINT_AXIS].correct == 1

    def test_partially_judged_row_counts_per_axis(self) -> None:
        """A half-filled row still contributes the axes it answered, and only those."""
        rows = [{"rid": "#9:1", "tier": "frame", "judge_span": "1", "judge_predicate": "0"}]
        frame = {tier.tier: tier for tier in aggregate(rows)}["frame"]
        assert frame.axes["span"].judged == 1
        assert frame.axes["direction"].judged == 0
        assert frame.axes[JOINT_AXIS].judged == 0

    def test_every_tier_is_present_even_with_no_rows(self) -> None:
        tiers = aggregate([_row("frame", "1111")])
        assert [tier.tier for tier in tiers] == list(TIERS)
        wordnet = {tier.tier: tier for tier in tiers}["wordnet"]
        assert wordnet.sampled == 0
        assert wordnet.axes["span"].precision is None

    def test_unknown_tier_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown tier"):
            aggregate([_row("frames", "1111")])

    def test_evidence_less_rows_are_counted_not_hidden(self) -> None:
        """An edge with no evidence chunk id cannot be judged — that is a finding."""
        judged = _row("lemma", "1111")
        blind = _row("lemma", "1111")
        blind["evidence_chunk_ids"] = ""
        lemma = {tier.tier: tier for tier in aggregate([judged, blind])}["lemma"]
        assert lemma.sampled == 2
        assert lemma.no_evidence == 1
        assert any("no evidence chunk id" in note for note in caveats(aggregate([judged, blind])))

    def test_populations_are_unknown_not_zero_without_a_manifest(self) -> None:
        tiers = aggregate([_row("frame", "1111")])
        assert tiers[0].population is None
        with_manifest = aggregate([_row("frame", "1111")], {"frame": 2472})
        assert with_manifest[0].population == 2472


class TestVerifierScore:
    def test_mean_median_and_null_count(self) -> None:
        rows = [
            _row("lemma", "1111", "0.60"),
            _row("lemma", "1111", "0.80"),
            _row("lemma", "1111", ""),
        ]
        lemma = {tier.tier: tier for tier in aggregate(rows)}["lemma"]
        assert lemma.verifier_scored == 2
        assert lemma.verifier_null == 1
        assert lemma.verifier_mean == pytest.approx(0.70)
        assert lemma.verifier_median == pytest.approx(0.70)

    def test_never_scored_is_not_a_zero(self) -> None:
        """NULL must not drag the mean down — 'unscored' is not 'scored 0.0'."""
        rows = [_row("lemma", "1111", "0.90"), _row("lemma", "1111", "")]
        lemma = {tier.tier: tier for tier in aggregate(rows)}["lemma"]
        assert lemma.verifier_mean == pytest.approx(0.90)

    def test_all_null(self) -> None:
        lemma = {tier.tier: tier for tier in aggregate([_row("lemma", "1111")])}["lemma"]
        assert lemma.verifier_mean is None
        assert lemma.verifier_median is None
        assert lemma.verifier_null == 1


class TestVerifierCoverageCaveat:
    """The monitor covers ~10% of the graph: `RelationVerifierGate` runs only in the
    relex write path, so the frame projection is structurally unscorable."""

    def test_blind_tier_is_named(self) -> None:
        rows = [_row("frame", "1111"), _row("ontology", "1111", "0.6")]
        notes = caveats(aggregate(rows))
        assert any("structurally cannot score this tier" in note for note in notes)

    def test_coverage_line_names_both_sides(self) -> None:
        rows = [_row("frame", "1111"), _row("ontology", "1111", "0.6")]
        coverage = next(note for note in caveats(aggregate(rows)) if "coverage is PARTIAL" in note)
        assert "`ontology`" in coverage and "`frame`" in coverage

    def test_fires_on_an_unjudged_sheet(self) -> None:
        """Coverage does not depend on anyone having judged anything; an early
        `continue` in caveats() once swallowed this on the baseline sheet."""
        rows = [{"rid": "#9:1", "tier": "frame", "evidence_chunk_ids": "abc"}]
        notes = caveats(aggregate(rows, {"frame": 2431}))
        assert any("structurally cannot score this tier" in note for note in notes)

    def test_no_caveat_when_every_sampled_tier_is_scored(self) -> None:
        notes = caveats(aggregate([_row("ontology", "1111", "0.6")]))
        assert not any("coverage is PARTIAL" in note for note in notes)


class TestCaveats:
    def test_thin_sample_is_called_out(self) -> None:
        rows = [_row("frame", "1111") for _ in range(10)]
        notes = caveats(aggregate(rows, {"frame": 2472}))
        assert any("n=10" in note and "pp" in note for note in notes)

    def test_absent_tier_says_why(self) -> None:
        notes = caveats(aggregate([_row("frame", "1111")], {"frame": 10, "wordnet": 0}))
        assert any("`wordnet`" in note and "nothing to audit" in note for note in notes)

    def test_unjudged_sheet_says_so_instead_of_reporting_zero_precision(self) -> None:
        rows = [{"rid": "#9:1", "tier": "frame"}]
        notes = caveats(aggregate(rows, {"frame": 10}))
        assert any("none judged yet" in note for note in notes)


class TestRender:
    def test_markdown_covers_every_tier_and_axis(self) -> None:
        rows = [_row("frame", "1111", "0.7"), _row("lemma", "1011", "")]
        report = build_report(
            aggregate(rows, {"frame": 2472, "wordnet": 0, "ontology": 99, "lemma": 166}),
            {"sheet": "s.csv", "database": "mem_x", "seed": 0, "generated_at": "now"},
        )
        markdown = render_markdown(report)
        for tier in TIERS:
            assert f"`{tier}`" in markdown
        for axis in (*AXES, JOINT_AXIS):
            assert axis in markdown
        assert "verifier_score" in markdown
        assert "mem_x" in markdown

    def test_report_is_json_serialisable(self) -> None:
        import json

        report = build_report(aggregate([_row("frame", "1111")]), {"sheet": "s.csv"})
        assert json.loads(json.dumps(report))["tiers"][0]["tier"] == "frame"

    def test_sheet_columns_carry_the_judgement_axes(self) -> None:
        assert set(JUDGEMENT_COLUMNS) <= set(SHEET_COLUMNS)
        assert "evidence_text" in SHEET_COLUMNS
        assert "attributed_to" in SHEET_COLUMNS

    def test_sheet_columns_are_unique(self) -> None:
        """`predicate` is both an axis and a REL property; a collision would silently
        drop one of them from the sheet, and the axis would score as unjudged forever."""
        assert len(set(SHEET_COLUMNS)) == len(SHEET_COLUMNS)
        assert "predicate" in SHEET_COLUMNS
        assert "judge_predicate" in SHEET_COLUMNS


class TestCompare:
    """`compare` is the missing piece behind #232: given two reports it must say,
    per tier and axis, whether precision moved and whether that move is bigger
    than the sampling noise — without recomputing any statistics itself."""

    def test_delta_arithmetic_matches_the_two_reports_own_precisions(self) -> None:
        # span verdict is the first character of each verdict string.
        baseline_rows = [_row("frame", v) for v in ("0111", "0111", "1111", "1111")]
        current_rows = [_row("frame", v) for v in ("1111", "1111", "1111", "0111")]
        baseline = build_report(aggregate(baseline_rows), {"sheet": "old.csv"})
        current = build_report(aggregate(current_rows), {"sheet": "new.csv"})

        deltas = compare(baseline, current)
        by_key = {(entry["tier"], entry["axis"]): entry for entry in deltas}

        span = by_key[("frame", "span")]
        baseline_precision = baseline["tiers"][0]["precision"]["span"]["precision"]
        current_precision = current["tiers"][0]["precision"]["span"]["precision"]
        assert span["baseline"] == baseline_precision
        assert span["current"] == current_precision
        assert span["delta_points"] == pytest.approx((current_precision - baseline_precision) * 100)

        # `wordnet` has no rows on either side: nothing to compare, not a zero.
        unjudged = by_key[("wordnet", "span")]
        assert unjudged["baseline"] is None
        assert unjudged["current"] is None
        assert unjudged["delta_points"] is None

    def test_entries_cover_every_tier_and_report_axis_in_order(self) -> None:
        baseline = build_report(aggregate([_row("frame", "1111")]), {"sheet": "old.csv"})
        current = build_report(aggregate([_row("frame", "1111")]), {"sheet": "new.csv"})
        deltas = compare(baseline, current)
        expected_keys = [(tier, axis) for tier in TIERS for axis in (*AXES, JOINT_AXIS)]
        assert [(entry["tier"], entry["axis"]) for entry in deltas] == expected_keys

    def test_small_change_over_few_rows_is_flagged_as_noise(self) -> None:
        """Wide, overlapping Wilson intervals: the two point estimates differ but
        the data cannot tell the difference from sampling noise."""
        baseline_low, baseline_high = wilson_interval(5, 5)
        current_low, current_high = wilson_interval(4, 5)
        assert (
            baseline_low <= current_high and current_low <= baseline_high
        )  # self-evidently overlapping

        baseline_rows = [_row("frame", "1111") for _ in range(5)]
        current_rows = [_row("frame", "1111") for _ in range(4)] + [_row("frame", "0111")]
        baseline = build_report(aggregate(baseline_rows), {"sheet": "old.csv"})
        current = build_report(aggregate(current_rows), {"sheet": "new.csv"})

        span = next(
            e for e in compare(baseline, current) if e["tier"] == "frame" and e["axis"] == "span"
        )
        assert span["overlapping"] is True

    def test_large_change_over_enough_rows_is_not_noise(self) -> None:
        """Enough rows that the intervals no longer touch: a real, measurable move."""
        baseline_low, baseline_high = wilson_interval(0, 20)
        current_low, current_high = wilson_interval(20, 20)
        assert baseline_high < current_low  # self-evidently disjoint

        baseline_rows = [_row("frame", "0111") for _ in range(20)]
        current_rows = [_row("frame", "1111") for _ in range(20)]
        baseline = build_report(aggregate(baseline_rows), {"sheet": "old.csv"})
        current = build_report(aggregate(current_rows), {"sheet": "new.csv"})

        span = next(
            e for e in compare(baseline, current) if e["tier"] == "frame" and e["axis"] == "span"
        )
        assert span["overlapping"] is False


class TestCompareShapeMismatch:
    """A comparison across incompatible report shapes must refuse rather than
    silently pair up the wrong tiers or axes."""

    def test_differing_axes_lists_raise(self) -> None:
        current = build_report(aggregate([_row("frame", "1111")]), {"sheet": "new.csv"})
        baseline = dict(current)
        baseline["axes"] = current["axes"][:-1]  # drops all_four
        with pytest.raises(ValueError, match="axes"):
            compare(baseline, current)

    def test_differing_tier_sequence_raises(self) -> None:
        current = build_report(aggregate([_row("frame", "1111")]), {"sheet": "new.csv"})
        baseline = dict(current)
        baseline["tiers"] = list(reversed(current["tiers"]))
        with pytest.raises(ValueError, match="tiers"):
            compare(baseline, current)


class TestRenderMarkdownBaseline:
    """`render_markdown` grows a baseline section only when the report carries one;
    every report produced before #232 must keep rendering exactly as it did."""

    def test_baseline_section_present_when_report_carries_baseline(self) -> None:
        baseline_rows = [_row("frame", v) for v in ("0111", "0111", "1111", "1111")]
        current_rows = [_row("frame", v) for v in ("1111", "1111", "1111", "0111")]
        baseline_report = build_report(aggregate(baseline_rows), {"sheet": "old-report.json"})
        current_report = build_report(aggregate(current_rows), {"sheet": "new-report.json"})
        deltas = compare(baseline_report, current_report)

        report = {**current_report, "baseline": {"report": "old-report.json", "deltas": deltas}}
        markdown = render_markdown(report)

        assert "baseline" in markdown.lower()
        assert any(
            line.startswith("##") and "baseline" in line.lower() for line in markdown.splitlines()
        )
        assert "old-report.json" in markdown
        # the frame/span delta is (0.75 - 0.5) * 100 == 25.0 points; the rendered
        # section must show that number somewhere, however it formats the sign.
        assert "25.0" in markdown

    def test_no_baseline_section_when_report_has_no_baseline_key(self) -> None:
        report = build_report(aggregate([_row("frame", "1111")]), {"sheet": "s.csv"})
        markdown = render_markdown(report)
        assert not any(
            line.startswith("##") and "baseline" in line.lower() for line in markdown.splitlines()
        )
