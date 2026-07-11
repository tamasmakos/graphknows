"""Generic report writers (JSON + CSV + Markdown) for all benchmarks.

Headline metric: **accuracy** — the fraction of cases whose LLM-judge score is
≥ 0.5 (binary judges emit exactly 0.0/1.0, rubric judges a mean in [0, 1]).
``avg_judge_score`` is the mean raw score. Per-category breakdowns are built
dynamically from whatever ``case.category`` values appear.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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
