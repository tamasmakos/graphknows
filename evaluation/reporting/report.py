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
    "judge_correct",
    "gold_max_token_overlap",
    "gold_reachable_at_20",
    "avg_hit_score",
    "recall_hit_count",
    "ltm_graph_hit_count",
    # DSPy semantic metrics (populated when EVAL_DSPY_METRICS=1)
    "semantic_f1",
    "complete_and_grounded",
    "completeness",
    "groundedness",
    "ingested_chunks",
    "ingested_entities",
    "ltm_entity_count",
    "flush_nodes",
    "flush_edges",
    # OpenRouter usage (per question)
    "gen_prompt_tokens",
    "gen_completion_tokens",
    "gen_total_tokens",
    "gen_cost_usd",
    # OpenRouter usage (ingestion, llm_assisted only)
    "ingest_prompt_tokens",
    "ingest_completion_tokens",
    "ingest_cost_usd",
)


def summarize(cases: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case scores and metrics into a run summary.

    Reports two headline numbers: ``avg_score`` (lexical token-F1) and
    ``judge_accuracy`` (semantic LLM-judge correctness — the leaderboard metric).
    """
    scores = [case.score for case in cases]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    by_category: dict[str, list[float]] = {}
    judge_by_category: dict[str, list[float]] = {}
    for case in cases:
        cat = case.category or "uncategorized"
        by_category.setdefault(cat, []).append(case.score)
        judge_by_category.setdefault(cat, []).append(
            1.0 if case.metrics.get("judge_correct") else 0.0
        )

    judge_vals = [1.0 if case.metrics.get("judge_correct") else 0.0 for case in cases]
    judge_accuracy = sum(judge_vals) / len(judge_vals) if judge_vals else 0.0

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
        "judge_accuracy": judge_accuracy,
        "avg_score_by_category": {
            cat: sum(vals) / len(vals) for cat, vals in sorted(by_category.items())
        },
        "judge_accuracy_by_category": {
            cat: sum(vals) / len(vals) for cat, vals in sorted(judge_by_category.items())
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
        f"- Mean token-F1 (lexical): `{summary['avg_score']:.3f}`",
        f"- Judge accuracy (semantic): `{summary.get('judge_accuracy', 0.0):.3f}`",
        "",
        "## Score by category (token-F1 | judge)",
        "",
    ]
    judge_cat = summary.get("judge_accuracy_by_category", {})
    for cat, val in summary["avg_score_by_category"].items():
        lines.append(f"- `{cat}`: F1 `{val:.3f}`  |  judge `{judge_cat.get(cat, 0.0):.3f}`")
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
