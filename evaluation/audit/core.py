"""Pure logic for the stratified triplet audit: tiers, sampling, Wilson, aggregation.

Why this exists: the system is generation-bound and ``evidence_recall`` provably
cannot see graph-quality work — a run that added an entire symbolic plane (1240
frame instances gaining polarity/modality/epistemic, 2735 REL edges gaining a
stable predicate id) reproduced ``evidence_recall_in_context=0.9255`` to four
decimals. Triplet precision therefore needs an instrument that never touches the
retriever: a human judges a stratified sample of REL edges against their evidence
text, and this module turns those judgements into per-tier precision.

Everything here is a function of its arguments — no database, no filesystem — so
the stratification and the statistics are testable without a graph.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The stratification axis is `canonical_predicate`, whose prefix encodes how the
# predicate is anchored. Report order is fixed so two audits compare row-by-row;
# `wordnet` is listed even though no channel emits `wn30:` yet, so a report says
# "zero" out loud instead of silently dropping the tier.
TIERS: tuple[str, ...] = ("frame", "wordnet", "ontology", "lemma", "unclassified")

_TIER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("fn:", "frame"),
    ("wn30:", "wordnet"),
    ("lemma:", "lemma"),
)

# The four binary judgements a human makes per edge, plus the derived conjunction.
AXES: tuple[str, ...] = ("span", "predicate", "direction", "polarity_modality")
JOINT_AXIS = "all_four"

# Judgement columns are prefixed, because `predicate` is BOTH an axis and a REL
# property: an unprefixed axis column collides with the data column of the same
# name and `csv.DictWriter` keeps only one of them, so a whole axis would ship
# blank and score as unjudged forever.
JUDGEMENT_COLUMNS: tuple[str, ...] = tuple(f"judge_{axis}" for axis in AXES)

# Columns of the judgement sheet. The four judgement columns ship blank — a human
# fills them in — and `notes` is free text that nothing parses.
SHEET_COLUMNS: tuple[str, ...] = (
    "rid",
    "tier",
    "canonical_predicate",
    "predicate",
    "head",
    "tail",
    "head_role",
    "tail_role",
    "frame",
    "polarity",
    "modality",
    "epistemic",
    "hedged",
    "attributed_to",
    "anchor",
    "reified",
    "confidence",
    "verifier_score",
    "evidence_chunk_ids",
    "evidence_text",
    *JUDGEMENT_COLUMNS,
    "notes",
)

_TRUE_WORDS = frozenset({"1", "y", "yes", "true", "t", "ok"})
_FALSE_WORDS = frozenset({"0", "n", "no", "false", "f"})

# Below this many judged rows the Wilson interval is wider than most of the
# effects anyone wants to detect, so the report says so rather than printing a
# three-decimal precision that reads like a measurement.
THIN_SAMPLE = 50


def tier_of(canonical_predicate: str | None) -> str:
    """Map a ``canonical_predicate`` onto its audit tier.

    Args:
        canonical_predicate: The REL edge's stable predicate id, or ``None``.

    Returns:
        One of :data:`TIERS`. An unprefixed non-empty value is an ontology
        property label/IRI; an empty one is ``"unclassified"`` rather than being
        quietly counted as an ontology property.
    """
    value = (canonical_predicate or "").strip()
    for prefix, tier in _TIER_PREFIXES:
        if value.startswith(prefix):
            return tier
    return "ontology" if value else "unclassified"


def tier_census(edges: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count edges per tier, keeping every tier in :data:`TIERS` including empty ones."""
    counts = Counter(tier_of(edge.get("canonical_predicate")) for edge in edges)
    return {tier: counts.get(tier, 0) for tier in TIERS}


def allocate(populations: Mapping[str, int], total: int) -> dict[str, int]:
    """Split *total* draws as evenly as possible over tiers, capped by population.

    A tier that cannot fill its equal share (or has no edges at all) hands the
    remainder back to the tiers that can, so the sheet holds *total* rows whenever
    the graph has that many edges.

    Args:
        populations: Tier -> number of edges available.
        total: How many edges to draw in all.

    Returns:
        Tier -> number of edges to draw, one key per key of *populations*.
    """
    quota = dict.fromkeys(populations, 0)
    remaining = max(0, total)
    open_tiers = {tier for tier, n in populations.items() if n > 0}
    while remaining > 0 and open_tiers:
        share = max(1, remaining // len(open_tiers))
        for tier in sorted(open_tiers):
            if remaining <= 0:
                break
            take = min(share, populations[tier] - quota[tier], remaining)
            quota[tier] += take
            remaining -= take
        open_tiers = {tier for tier in open_tiers if quota[tier] < populations[tier]}
    return quota


def stratified_sample(
    edges: Sequence[Mapping[str, Any]], total: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Draw *total* edges spread as evenly as possible over the tiers.

    Deterministic given *seed* and the same edge set: each tier's pool is sorted
    by ``rid`` before sampling, so a re-run reproduces the sheet exactly.

    Args:
        edges: REL edge rows; each needs ``canonical_predicate`` and ``rid``.
        total: Sample size across all tiers.
        seed: PRNG seed.

    Returns:
        ``(sampled_rows, census, quota)`` — the rows each carry an added ``tier``
        key and are ordered by (tier, rid).
    """
    census = tier_census(edges)
    quota = allocate(census, total)
    pools: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    for edge in edges:
        row = dict(edge)
        row["tier"] = tier_of(row.get("canonical_predicate"))
        pools[row["tier"]].append(row)

    rng = random.Random(seed)  # audit sampling, not cryptography
    picked: list[dict[str, Any]] = []
    for tier in TIERS:
        pool = sorted(pools[tier], key=lambda row: str(row.get("rid", "")))
        picked.extend(rng.sample(pool, quota[tier]))
    picked.sort(key=lambda row: (TIERS.index(str(row["tier"])), str(row.get("rid", ""))))
    return picked, census, quota


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at the default *z*).

    Wilson rather than the normal approximation because the audit lives at small
    *n* and high *p*, exactly where the normal interval runs off the end of [0, 1].

    Args:
        successes: Number of correct judgements.
        n: Number of judgements made.
        z: Standard-normal quantile; 1.96 is 95%.

    Returns:
        ``(low, high)``. ``n == 0`` returns ``(0.0, 1.0)`` — no observations means
        no knowledge, not a point estimate of zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def parse_verdict(raw: str | None) -> bool | None:
    """Read one judgement cell from the sheet.

    Args:
        raw: The cell text.

    Returns:
        ``True``/``False``, or ``None`` when the cell is blank (not yet judged).

    Raises:
        ValueError: The cell holds something that is neither blank nor a verdict.
            A typo must not be silently scored as a failure.
    """
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value in _TRUE_WORDS:
        return True
    if value in _FALSE_WORDS:
        return False
    raise ValueError(f"unreadable judgement {raw!r} - use 1/0 (or y/n), blank for unjudged")


def _parse_score(raw: str | None) -> float | None:
    """Read a ``verifier_score`` cell; blank is a real value ('never scored')."""
    value = (raw or "").strip()
    if not value:
        return None
    return float(value)


@dataclass(frozen=True)
class AxisPrecision:
    """Precision on one judging axis within one tier."""

    axis: str
    judged: int
    correct: int
    precision: float | None
    ci_low: float
    ci_high: float

    @property
    def half_width(self) -> float:
        """Half the 95% interval, in proportion units."""
        return (self.ci_high - self.ci_low) / 2.0


@dataclass(frozen=True)
class TierPrecision:
    """Everything the report says about one ``canonical_predicate`` tier."""

    tier: str
    population: int | None
    sampled: int
    #: Sampled rows carrying no evidence chunk id at all. A human cannot judge
    #: these, so they are reported rather than counted as passes or failures.
    no_evidence: int
    axes: dict[str, AxisPrecision]
    verifier_scored: int
    verifier_null: int
    verifier_mean: float | None
    verifier_median: float | None


def _axis_precision(axis: str, verdicts: Sequence[bool]) -> AxisPrecision:
    judged = len(verdicts)
    correct = sum(1 for verdict in verdicts if verdict)
    low, high = wilson_interval(correct, judged)
    return AxisPrecision(
        axis=axis,
        judged=judged,
        correct=correct,
        precision=(correct / judged) if judged else None,
        ci_low=low,
        ci_high=high,
    )


def _tier_precision(
    tier: str, rows: Sequence[Mapping[str, str]], population: int | None
) -> TierPrecision:
    verdicts: dict[str, list[bool]] = {axis: [] for axis in AXES}
    joint: list[bool] = []
    for row in rows:
        judged = {axis: parse_verdict(row.get(f"judge_{axis}")) for axis in AXES}
        for axis, verdict in judged.items():
            if verdict is not None:
                verdicts[axis].append(verdict)
        if all(verdict is not None for verdict in judged.values()):
            joint.append(all(judged.values()))

    scores = [
        score for row in rows if (score := _parse_score(row.get("verifier_score"))) is not None
    ]
    no_evidence = sum(1 for row in rows if not (row.get("evidence_chunk_ids") or "").strip())
    axes = {axis: _axis_precision(axis, verdicts[axis]) for axis in AXES}
    axes[JOINT_AXIS] = _axis_precision(JOINT_AXIS, joint)
    return TierPrecision(
        tier=tier,
        population=population,
        sampled=len(rows),
        no_evidence=no_evidence,
        axes=axes,
        verifier_scored=len(scores),
        verifier_null=len(rows) - len(scores),
        verifier_mean=statistics.fmean(scores) if scores else None,
        verifier_median=statistics.median(scores) if scores else None,
    )


def aggregate(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int] | None = None,
) -> list[TierPrecision]:
    """Turn a judged sheet into one :class:`TierPrecision` per tier.

    Args:
        rows: Judged sheet rows (``csv.DictReader`` output).
        populations: Tier -> edges in the graph at sample time, from the sample
            manifest. ``None`` leaves populations unknown rather than guessing 0.

    Returns:
        One entry per tier in :data:`TIERS` order, including tiers with no rows.

    Raises:
        ValueError: A row carries a tier name this audit does not know.
    """
    by_tier: dict[str, list[Mapping[str, str]]] = {tier: [] for tier in TIERS}
    for row in rows:
        tier = (row.get("tier") or "").strip() or "unclassified"
        if tier not in by_tier:
            raise ValueError(f"row {row.get('rid')!r} carries unknown tier {tier!r}")
        by_tier[tier].append(row)
    return [_tier_precision(tier, by_tier[tier], (populations or {}).get(tier)) for tier in TIERS]


@dataclass(frozen=True)
class FactPrecision:
    """One run's fact precision: the audit's tiers pooled into a single number.

    The audit itself is corpus-agnostic — it judges REL edges against their
    evidence chunks, whatever the chunks were made of — so the run-level metric is
    keyed by *row* rather than assuming the dialogue corpus, and a row with no
    judged sheet says why instead of being omitted (contracts/panel-report.md).
    """

    row: str
    judged: int
    correct: int
    precision: float | None
    ci_low: float
    ci_high: float
    by_tier: dict[str, float | None]
    not_judged: str | None

    def to_record(self) -> dict[str, Any]:
        """The contract's per-run ``fact_precision`` block."""
        return {
            "row": self.row,
            "judged": self.judged,
            "correct": self.correct,
            "precision": None if self.precision is None else round(self.precision, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "by_tier": {
                tier: None if value is None else round(value, 4)
                for tier, value in self.by_tier.items()
            },
            "not_judged": self.not_judged,
        }


def fact_precision(row: str, tiers: Sequence[TierPrecision]) -> FactPrecision:
    """Pool the joint axis over all tiers into the run's fact-precision metric.

    Args:
        row: The panel row this audit belongs to (``locomo``, ``repo``, ...).
        tiers: :func:`aggregate` output, or empty when the row has no audit sheet.

    Returns:
        A :class:`FactPrecision`. With nothing judged, ``precision`` is ``None``
        and ``not_judged`` carries the reason — the run still reports the metric.
    """
    joint = [tier.axes[JOINT_AXIS] for tier in tiers]
    judged = sum(axis.judged for axis in joint)
    correct = sum(axis.correct for axis in joint)
    low, high = wilson_interval(correct, judged)
    reason = None
    if not tiers:
        reason = f"no audit sheet for row {row!r}"
    elif not judged:
        reason = f"audit sheet for row {row!r} has no judged rows"
    return FactPrecision(
        row=row,
        judged=judged,
        correct=correct,
        precision=(correct / judged) if judged else None,
        ci_low=low,
        ci_high=high,
        by_tier={tier.tier: tier.axes[JOINT_AXIS].precision for tier in tiers},
        not_judged=reason,
    )


def caveats(tiers: Sequence[TierPrecision]) -> list[str]:
    """Sample-size honesty, computed from the sheet rather than pasted boilerplate.

    A stratified 150 spread over four tiers is ~37 per tier, where the 95% Wilson
    interval is roughly +/-15pp near p=0.5. Printing that next to the number is
    the difference between a direction and a false measurement.
    """
    notes: list[str] = []
    covered = [tier.tier for tier in tiers if tier.verifier_scored]
    blind = [tier.tier for tier in tiers if tier.sampled and not tier.verifier_scored]
    if blind:
        notes.append(
            "verifier_score coverage is PARTIAL: scored on "
            f"{', '.join(f'`{name}`' for name in covered) or 'no tier'}, never scored on "
            f"{', '.join(f'`{name}`' for name in blind)}. Read no mean below as a "
            "whole-graph number."
        )
    for tier in tiers:
        if tier.population == 0:
            notes.append(f"`{tier.tier}`: no edges of this tier in the graph — nothing to audit.")
            continue
        if tier.sampled == 0:
            notes.append(f"`{tier.tier}`: not represented in this sheet.")
            continue
        joint = tier.axes[JOINT_AXIS]
        if tier.no_evidence:
            notes.append(
                f"`{tier.tier}`: {tier.no_evidence} of {tier.sampled} sampled edges carry no "
                "evidence chunk id, so no human can judge them — an evidence gap, not a "
                "precision result."
            )
        if tier.sampled and not tier.verifier_scored:
            notes.append(
                f"`{tier.tier}`: verifier_score is NULL on every sampled edge - "
                "`RelationVerifierGate` runs only in the relex write path, so it "
                "structurally cannot score this tier. Its mean is not a health number here."
            )
        if joint.judged == 0:
            notes.append(f"`{tier.tier}`: {tier.sampled} rows sampled, none judged yet.")
            continue
        if joint.judged < THIN_SAMPLE:
            notes.append(
                f"`{tier.tier}`: n={joint.judged} — the 95% interval is "
                f"+/-{joint.half_width * 100:.0f}pp. Read it as a direction, not a measurement."
            )
    return notes


def build_report(tiers: Sequence[TierPrecision], meta: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the JSON-able audit report from per-tier results and run metadata.

    ``meta["row"]`` names the panel row being audited; the report carries its
    pooled ``fact_precision`` block so any run, dialogue or not, reports one.
    """
    row = str(dict(meta).get("row") or "unknown")
    return {
        **dict(meta),
        "axes": [*AXES, JOINT_AXIS],
        "fact_precision": fact_precision(row, tiers).to_record(),
        "caveats": caveats(tiers),
        "tiers": [
            {
                "tier": tier.tier,
                "population": tier.population,
                "sampled": tier.sampled,
                "no_evidence": tier.no_evidence,
                "verifier_score": {
                    "scored": tier.verifier_scored,
                    "null": tier.verifier_null,
                    "mean": (
                        round(tier.verifier_mean, 4) if tier.verifier_mean is not None else None
                    ),
                    "median": (
                        round(tier.verifier_median, 4) if tier.verifier_median is not None else None
                    ),
                },
                "precision": {
                    axis: {
                        "judged": result.judged,
                        "correct": result.correct,
                        "precision": (
                            round(result.precision, 4) if result.precision is not None else None
                        ),
                        "ci_low": round(result.ci_low, 4),
                        "ci_high": round(result.ci_high, 4),
                    }
                    for axis, result in tier.axes.items()
                },
            }
            for tier in tiers
        ],
    }


def compare(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-tier, per-axis precision deltas between two reports (issue #232).

    Reads point estimates and Wilson intervals straight out of both reports'
    ``tiers[].precision``; never recomputes a Wilson interval, since
    :func:`wilson_interval` already produced the ones serialised there.

    Args:
        baseline: An earlier report dict, as produced by :func:`build_report`.
        current: The report being compared against it.

    Returns:
        One dict per (tier, axis) pair, in the reports' own tier and axis order,
        with keys ``tier``, ``axis``, ``baseline``, ``current``, ``delta_points``,
        ``overlapping``.

    Raises:
        ValueError: The two reports' ``axes`` or tier-name sequences differ, or
            either is missing ``axes``/``tiers``.
    """
    if "axes" not in baseline or "axes" not in current:
        raise ValueError("both reports must carry an 'axes' list")
    if baseline["axes"] != current["axes"]:
        raise ValueError(f"axes differ: baseline={baseline['axes']!r} current={current['axes']!r}")
    if "tiers" not in baseline or "tiers" not in current:
        raise ValueError("both reports must carry a 'tiers' list")
    baseline_tier_names = [tier["tier"] for tier in baseline["tiers"]]
    current_tier_names = [tier["tier"] for tier in current["tiers"]]
    if baseline_tier_names != current_tier_names:
        raise ValueError(
            f"tiers differ: baseline={baseline_tier_names!r} current={current_tier_names!r}"
        )

    axes = current["axes"]
    deltas: list[dict[str, Any]] = []
    for baseline_tier, current_tier in zip(baseline["tiers"], current["tiers"], strict=True):
        tier_name = current_tier["tier"]
        for axis in axes:
            baseline_entry = baseline_tier["precision"][axis]
            current_entry = current_tier["precision"][axis]
            baseline_precision = baseline_entry["precision"]
            current_precision = current_entry["precision"]
            if baseline_precision is None or current_precision is None:
                delta_points = None
                overlapping = False
            else:
                delta_points = round((current_precision - baseline_precision) * 100, 1)
                overlapping = (
                    baseline_entry["ci_low"] <= current_entry["ci_high"]
                    and current_entry["ci_low"] <= baseline_entry["ci_high"]
                )
            deltas.append(
                {
                    "tier": tier_name,
                    "axis": axis,
                    "baseline": baseline_precision,
                    "current": current_precision,
                    "delta_points": delta_points,
                    "overlapping": overlapping,
                }
            )
    return deltas


def _cell(entry: Mapping[str, Any]) -> str:
    """One precision cell: the point estimate with its interval, or an em dash."""
    if entry["judged"] == 0 or entry["precision"] is None:
        return "—"
    return f"{entry['precision']:.2f} [{entry['ci_low']:.2f}-{entry['ci_high']:.2f}]"


def _fact_precision_cell(record: Mapping[str, Any]) -> str:
    """The run's pooled fact precision, or the reason there is none."""
    if record["precision"] is None:
        return f"— ({record['not_judged']})"
    return (
        f"`{record['row']}` {record['precision']:.2f} "
        f"[{record['ci_low']:.2f}-{record['ci_high']:.2f}] over {record['judged']} judged rows"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the audit report as Markdown."""
    axes = list(report["axes"])
    lines = [
        "# Stratified Triplet Audit",
        "",
        f"- Sheet: `{report.get('sheet', '?')}`",
        f"- Database: `{report.get('database', '?')}`",
        f"- Seed: `{report.get('seed', '?')}`",
        f"- Generated: `{report.get('generated_at', '?')}`",
        f"- Fact precision: {_fact_precision_cell(report['fact_precision'])}",
        "",
        "Precision measured directly on the graph — no retrieval in the loop.",
        "Each cell is `precision [95% Wilson interval]` over the rows judged on that axis.",
        "",
        "## Precision by tier",
        "",
        "| Tier | population | sampled | no evidence | judged | " + " | ".join(axes) + " |",
        "|------|------------|---------|-------------|--------|"
        + "|".join(["---"] * len(axes))
        + "|",
    ]
    for tier in report["tiers"]:
        precision = tier["precision"]
        population = "—" if tier["population"] is None else str(tier["population"])
        cells = " | ".join(_cell(precision[axis]) for axis in axes)
        lines.append(
            f"| `{tier['tier']}` | {population} | {tier['sampled']} | {tier['no_evidence']} | "
            f"{precision[JOINT_AXIS]['judged']} | {cells} |"
        )

    lines.extend(
        [
            "",
            "## verifier_score by tier",
            "",
            "The continuous monitor between audits: computed at flush and stored on every"
            " REL edge. A NULL means the verifier never scored that edge — deliberately"
            " distinguishable from a scored 0.0.",
            "",
            "| Tier | scored | NULL | mean | median |",
            "|------|--------|------|------|--------|",
        ]
    )
    for tier in report["tiers"]:
        score = tier["verifier_score"]
        mean = "—" if score["mean"] is None else f"{score['mean']:.3f}"
        median = "—" if score["median"] is None else f"{score['median']:.3f}"
        lines.append(
            f"| `{tier['tier']}` | {score['scored']} | {score['null']} | {mean} | {median} |"
        )

    baseline = report.get("baseline")
    if baseline is not None:
        by_tier: dict[str, dict[str, dict[str, Any]]] = {}
        for entry in baseline["deltas"]:
            by_tier.setdefault(entry["tier"], {})[entry["axis"]] = entry
        lines.extend(
            [
                "",
                "## Change from baseline",
                "",
                f"Compared against `{baseline['report']}`. Each cell is the precision change"
                " in points; `*` marks a cell whose Wilson intervals overlap the baseline's —"
                " not distinguishable from noise at this sample size.",
                "",
                "| Tier | " + " | ".join(axes) + " |",
                "|------|" + "|".join(["---"] * len(axes)) + "|",
            ]
        )
        for tier in report["tiers"]:
            tier_name = tier["tier"]
            cell_values: list[str] = []
            for axis in axes:
                entry = by_tier[tier_name][axis]
                if entry["delta_points"] is None:
                    cell_values.append("—")
                else:
                    marker = "*" if entry["overlapping"] else ""
                    cell_values.append(f"{entry['delta_points']:+.1f}{marker}")
            lines.append(f"| `{tier_name}` | " + " | ".join(cell_values) + " |")

    lines.extend(["", "## Read this before quoting a number", ""])
    lines.extend(f"- {note}" for note in report["caveats"])
    return "\n".join(lines).rstrip() + "\n"
