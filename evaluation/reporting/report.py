"""Report writers for LoCoMo evaluation runs (JSON + CSV + Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.datamodels import CaseResult, RunReport


@dataclass(frozen=True)
class ReportPaths:
    """Paths written for one evaluation run."""

    output_dir: Path
    json_path: Path
    csv_path: Path
    markdown_path: Path


# Per-case CSV columns — numeric metrics first (easy to load into pandas/R),
# then text fields last.
_CSV_METRIC_KEYS = (
    "gold_max_token_overlap",
    "gold_reachable_at_20",
    "avg_hit_score",
    "recall_hit_count",
    "ltm_graph_hit_count",
    "ingested_chunks",
    "ingested_entities",
    "ltm_entity_count",
    "flush_nodes",
    "flush_edges",
)


def summarize(cases: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case scores and metrics into a run summary."""
    scores = [case.score for case in cases]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    by_category: dict[str, list[float]] = {}
    for case in cases:
        by_category.setdefault(case.category or "uncategorized", []).append(case.score)

    metric_buckets: dict[str, list[float]] = {}
    for case in cases:
        for key, value in case.metrics.items():
            if isinstance(value, bool):
                metric_buckets.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                metric_buckets.setdefault(key, []).append(float(value))

    return {
        "case_count": len(cases),
        "avg_score": avg_score,
        "avg_score_by_category": {
            cat: sum(vals) / len(vals) for cat, vals in sorted(by_category.items())
        },
        "metric_summary": {
            key: sum(vals) / len(vals) for key, vals in sorted(metric_buckets.items()) if vals
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# LoCoMo Evaluation Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Mean token-F1: `{summary['avg_score']:.3f}`",
        "",
        "## Score by category",
        "",
    ]
    for cat, val in summary["avg_score_by_category"].items():
        lines.append(f"- `{cat}`: `{val:.3f}`")
    lines.extend(["", "## Aggregated metrics", ""])
    for key, val in summary["metric_summary"].items():
        lines.append(f"- `{key}`: `{val:.3f}`")
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        metrics = case.get("metrics", {})
        reachable = metrics.get("gold_reachable_at_20")
        reach_flag = (
            " ✅ reachable@20"
            if reachable is True
            else " ❌ NOT reachable@20"
            if reachable is False
            else ""
        )
        lines.append(f"### {case['case_id']}{reach_flag}")
        lines.append("")
        lines.append(case["question"])
        lines.append("")
        lines.append(f"**Gold**: `{case['gold']}`")
        lines.append(f"**Generated**: `{case['generated_answer']}`")
        lines.append(f"- token-F1: `{case['score']:.3f}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    header = [
        "case_id",
        "dialogue_id",
        "category",
        "score",
        *_CSV_METRIC_KEYS,
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
                    case["dialogue_id"],
                    case["category"],
                    f"{case['score']:.4f}",
                    *(metrics.get(k, "") for k in _CSV_METRIC_KEYS),
                    case["generated_answer"].replace("\n", " "),
                    case["gold"].replace("\n", " "),
                    case["question"].replace("\n", " "),
                ]
            )


def write_report(run: RunReport) -> ReportPaths:
    """Write JSON, CSV, and Markdown reports for a run into ``config.output_dir``."""
    out = run.config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    case_records = [case.to_record() for case in run.cases]
    report = {
        "benchmark": "locomo",
        "run_id": run.run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": run.config.model_dump(mode="json"),
        "summary": run.summary,
        "cases": case_records,
    }

    json_path = out / f"{run.run_id}-locomo.json"
    csv_path = out / f"{run.run_id}-locomo.csv"
    markdown_path = out / f"{run.run_id}-locomo.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, case_records)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    return ReportPaths(
        output_dir=out, json_path=json_path, csv_path=csv_path, markdown_path=markdown_path
    )
