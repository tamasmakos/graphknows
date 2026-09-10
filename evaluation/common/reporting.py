"""Generic report writers (JSON + CSV + Markdown) for all benchmarks.

Headline metric: **accuracy** — the fraction of cases whose LLM-judge score is
≥ 0.5 (binary judges emit exactly 0.0/1.0, rubric judges a mean in [0, 1]).
``avg_judge_score`` is the mean raw score. Per-category breakdowns are built
dynamically from whatever ``case.category`` values appear.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.common.datamodels import CaseResult, RunReport

PASS_THRESHOLD = 0.5


@dataclass(frozen=True)
class ReportPaths:
    """Paths written for one evaluation run."""

    output_dir: Path
    json_path: Path
    csv_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class PanelRow:
    """One benchmark row of a panel run (contracts/panel-report.md).

    ``dataclasses.asdict`` yields the contract's JSON shape. A row that could not
    be provisioned carries the reason in *not_run* and is never omitted.
    """

    row: str
    feeding_mode: str
    extractor: str
    accuracy: float = 0.0
    evidence_recall: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    n: int = 0
    not_run: str | None = None


@dataclass(frozen=True)
class InternalMetrics:
    """The once-per-run internal metrics (FR-040); ``asdict`` is the JSON shape."""

    identity: dict[str, float] = field(default_factory=dict)
    fact_precision: dict[str, float] = field(default_factory=dict)
    dead_weight: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineMetric:
    """Repeated runs of one metric. *median* is the headline, *band* the run noise."""

    runs: list[float]
    commit: str

    @property
    def median(self) -> float:
        return statistics.median(self.runs)

    @property
    def band(self) -> float:
        return max(self.runs) - min(self.runs)

    def to_record(self) -> dict[str, Any]:
        """The contract's ``median``/``band``/``runs``/``commit`` shape."""
        return {
            "median": self.median,
            "band": self.band,
            "runs": list(self.runs),
            "commit": self.commit,
        }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _accuracy(vals: list[float]) -> float:
    return _mean([1.0 if v >= PASS_THRESHOLD else 0.0 for v in vals])


def summarize(cases: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case scores and metrics into a run summary."""
    scores = [float(case.score) for case in cases]

    by_cat: dict[str, list[float]] = {}
    for case in cases:
        by_cat.setdefault(case.category or "uncategorized", []).append(float(case.score))

    # General numeric metric summary (averages of all numeric fields).
    metric_buckets: dict[str, list[float]] = {}
    for case in cases:
        for key, value in case.metrics.items():
            if isinstance(value, bool):
                metric_buckets.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                metric_buckets.setdefault(key, []).append(float(value))

    return {
        "case_count": len(cases),
        # Headline — LLM-judge accuracy (fraction of cases scoring ≥ 0.5)
        "accuracy": round(_accuracy(scores), 4),
        "avg_judge_score": round(_mean(scores), 4),
        "accuracy_by_category": {
            cat: round(_accuracy(vals), 4) for cat, vals in sorted(by_cat.items())
        },
        "judge_score_by_category": {
            cat: round(_mean(vals), 4) for cat, vals in sorted(by_cat.items())
        },
        "category_counts": {cat: len(vals) for cat, vals in sorted(by_cat.items())},
        "metric_summary": {
            key: round(_mean(vals), 4) for key, vals in sorted(metric_buckets.items()) if vals
        },
        "operational": _operational(cases, metric_buckets),
    }


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation, no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


def _operational(cases: list[CaseResult], buckets: dict[str, list[float]]) -> dict[str, Any]:
    """Cost / latency / behaviour, in the shape a README can quote.

    Kept apart from ``metric_summary`` (which is means over every numeric field)
    because the questions here have different shapes:

    * Latency covers RETRIEVAL ONLY, in percentiles (a mean hides the tail users
      feel). Retrieval is the only stage this system owns — local graph+vector
      work, no LLM. Generation/judge time is the configured provider's and would
      change under our name every time you swap models, so it is not reported;
      their COST is, because that is what a user actually spends.
    * Cost needs TOTALS, not means. ``ingest_cost_usd`` is the load-bearing one:
      in llm_free it is 0 by construction — the whole graph is built with no LLM
      call — so the only spend is at query time, against a corpus you ingest once.
    * Abstention is a rate, not a score: refusing to answer is a different defect
      from answering wrongly, and accuracy alone cannot tell them apart.
    """

    def _stat(key: str) -> dict[str, float] | None:
        vals = buckets.get(key)
        if not vals:
            return None
        return {
            "p50": round(_pct(vals, 50), 1),
            "p95": round(_pct(vals, 95), 1),
            "mean": round(_mean(vals), 1),
            "max": round(max(vals), 1),
        }

    latency = {name: stat for name in ("retrieve_ms",) if (stat := _stat(name)) is not None}

    def _total(key: str) -> float:
        return round(sum(buckets.get(key) or []), 4)

    n = len(cases) or 1
    # ingest_* is repeated onto every case of a unit, so summing it would multiply
    # by the question count. Take the distinct per-unit values instead.
    per_unit: dict[str, dict[str, float]] = {}
    for case in cases:
        unit = case.group_id or "?"
        m = case.metrics
        if unit not in per_unit:
            per_unit[unit] = {
                "ingest_cost_usd": float(m.get("ingest_cost_usd") or 0.0),
                "ingest_elapsed_s": float(m.get("ingest_elapsed_s") or 0.0),
                "ingested_chunks": float(m.get("ingested_chunks") or 0.0),
                "flush_nodes": float(m.get("flush_nodes") or 0.0),
            }
    gen_cost = _total("gen_cost_usd")
    ingest_cost = round(sum(u["ingest_cost_usd"] for u in per_unit.values()), 4)

    return {
        "latency_ms": latency,
        "cost_usd": {
            # 0.0 in llm_free: the graph is built with no LLM call at all.
            "ingest_total": ingest_cost,
            "query_total": gen_cost,
            "query_per_question": round(gen_cost / n, 6),
            "total": round(ingest_cost + gen_cost, 4),
        },
        "ingest": {
            "units": len(per_unit),
            "elapsed_s_total": round(sum(u["ingest_elapsed_s"] for u in per_unit.values()), 1),
            "chunks_total": int(sum(u["ingested_chunks"] for u in per_unit.values())),
            "graph_nodes_total": int(sum(u["flush_nodes"] for u in per_unit.values())),
        },
        "behaviour": {
            "abstention_rate": round(_mean(buckets.get("abstained") or [0.0]), 4),
            "avg_facts_supplied": round(_mean(buckets.get("fact_count") or [0.0]), 2),
            "avg_context_passages": round(_mean(buckets.get("context_passage_count") or [0.0]), 2),
        },
        "context": {
            "prompt_tokens_p50": round(_pct(buckets.get("gen_prompt_tokens") or [0.0], 50), 0),
            "prompt_tokens_p95": round(_pct(buckets.get("gen_prompt_tokens") or [0.0], 95), 0),
            "completion_tokens_p50": round(
                _pct(buckets.get("gen_completion_tokens") or [0.0], 50), 0
            ),
        },
    }


def _numeric_metric_keys(cases: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for case in cases:
        for key, value in case.get("metrics", {}).items():
            if isinstance(value, (bool, int, float)):
                keys.add(key)
    return sorted(keys)


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# {report['benchmark']} Evaluation Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Cases: `{summary['case_count']}`",
        "",
        "## Headline metric",
        "",
        f"**Accuracy** (LLM judge): `{summary.get('accuracy', 0.0):.3f}`  "
        f"_(mean judge score: `{summary.get('avg_judge_score', 0.0):.3f}`)_",
        "",
        "## Accuracy by category",
        "",
    ]
    counts = summary.get("category_counts", {})
    for cat, acc in summary.get("accuracy_by_category", {}).items():
        lines.append(f"- `{cat}`: `{acc:.3f}`  (n={counts.get(cat, 0)})")
    lines.extend(["", "## Aggregated metrics", ""])
    for key, val in summary.get("metric_summary", {}).items():
        lines.append(f"- `{key}`: `{val:.4f}`")
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        metrics = case.get("metrics", {})
        verdict = "✅" if case.get("score", 0.0) >= PASS_THRESHOLD else "❌"
        lines.append(f"### {case['case_id']} {verdict}")
        lines.append("")
        lines.append(case["question"])
        lines.append("")
        lines.append(f"**Gold**: `{case['gold']}`")
        lines.append(f"**Generated**: `{case['generated_answer']}`")
        lines.append(f"- judge score: `{case.get('score', 0.0):.3f}`")
        reasoning = metrics.get("judge_reasoning")
        if reasoning:
            lines.append(f"- judge reasoning: {reasoning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    metric_keys = _numeric_metric_keys(cases)
    header = [
        "case_id",
        "group_id",
        "category",
        "score",
        *metric_keys,
        "generated_answer",
        "gold",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for case in cases:
            metrics = case.get("metrics", {})
            writer.writerow(
                [
                    case["case_id"],
                    case["group_id"],
                    case["category"],
                    f"{case['score']:.4f}",
                    *(metrics.get(k, "") for k in metric_keys),
                    case["generated_answer"].replace("\n", " "),
                    case["gold"].replace("\n", " "),
                    case["question"].replace("\n", " "),
                ]
            )


def write_report(run: RunReport, output_dir: Path) -> ReportPaths:
    """Write JSON, CSV, and Markdown reports for a run into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    case_records = [case.to_record() for case in run.cases]
    report = {
        "benchmark": run.benchmark,
        "run_id": run.run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": run.config,
        "summary": run.summary,
        "cases": case_records,
    }

    json_path = output_dir / f"{run.run_id}-{run.benchmark}.json"
    csv_path = output_dir / f"{run.run_id}-{run.benchmark}.csv"
    markdown_path = output_dir / f"{run.run_id}-{run.benchmark}.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, case_records)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    return ReportPaths(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


def compare_modes(
    llm_free_cases: list[CaseResult],
    llm_assisted_cases: list[CaseResult],
) -> str:
    """Side-by-side per-category accuracy comparison table (Markdown)."""

    def _cat_acc(cases: list[CaseResult]) -> dict[str, float]:
        by_cat: dict[str, list[float]] = {}
        for case in cases:
            by_cat.setdefault(case.category or "uncategorized", []).append(float(case.score))
        return {cat: round(_accuracy(vals), 4) for cat, vals in sorted(by_cat.items())}

    free_acc = _cat_acc(llm_free_cases)
    asst_acc = _cat_acc(llm_assisted_cases)
    all_cats = sorted(set(free_acc) | set(asst_acc))

    free_overall = round(_accuracy([float(c.score) for c in llm_free_cases]), 4)
    asst_overall = round(_accuracy([float(c.score) for c in llm_assisted_cases]), 4)

    lines = [
        "# Accuracy Mode Comparison",
        "",
        "| Category | llm_free | llm_assisted | Δ (assisted - free) |",
        "|----------|----------|--------------|---------------------|",
    ]
    for cat in all_cats:
        f = free_acc.get(cat, 0.0)
        a = asst_acc.get(cat, 0.0)
        delta = a - f
        sign = "+" if delta >= 0 else ""
        lines.append(f"| `{cat}` | {f:.3f} | {a:.3f} | {sign}{delta:.3f} |")

    delta_overall = asst_overall - free_overall
    sign_overall = "+" if delta_overall >= 0 else ""
    lines.append(
        f"| **Overall** | **{free_overall:.3f}** | **{asst_overall:.3f}** "
        f"| **{sign_overall}{delta_overall:.3f}** |"
    )
    lines.append("")
    lines.append(f"_llm_free n={len(llm_free_cases)}  llm_assisted n={len(llm_assisted_cases)}_")
    return "\n".join(lines) + "\n"


EXTRACTOR_COLUMNS = ("local", "llm")
_PANEL_METRICS = ("accuracy", "evidence_recall")


def _panel_cell(row: PanelRow | None, metric: str) -> str:
    """One table cell: a score, the reason a row did not run, or an absent path."""
    if row is None:
        return "—"
    if row.not_run:
        return f"not run ({row.not_run})"
    return f"{float(getattr(row, metric)):.3f}"


def render_panel(rows: list[PanelRow]) -> str:
    """Panel table with the local and LLM extraction paths as separate columns (FR-043)."""
    by_row: dict[str, dict[str, PanelRow]] = {}
    for row in rows:
        by_row.setdefault(row.row, {})[row.extractor] = row

    header = ["Row", *(f"{ex} {m}" for ex in EXTRACTOR_COLUMNS for m in _PANEL_METRICS)]
    lines = [
        "# Panel",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, paths in sorted(by_row.items()):
        cells = [
            _panel_cell(paths.get(ex), metric)
            for ex in EXTRACTOR_COLUMNS
            for metric in _PANEL_METRICS
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
