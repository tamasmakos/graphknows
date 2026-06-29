"""Report writers for LoCoMo evaluation runs (JSON + CSV + Markdown).

Headline metric: mean DSPy SemanticF1 across all cases.
Secondary diagnostic: token-F1 (lexical, cheap, no LLM call).
Per-category SemanticF1 breakdown covers the five LoCoMo question types:
single_hop, multi_hop, temporal, open_domain, adversarial.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.datamodels import CaseResult, RunReport

# The five canonical LoCoMo question categories for the per-category breakdown.
LOCOMO_CATEGORIES = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
)


@dataclass(frozen=True)
class ReportPaths:
    """Paths written for one evaluation run."""

    output_dir: Path
    json_path: Path
    csv_path: Path
    markdown_path: Path


# Per-case CSV columns — numeric metrics first (easy to load into pandas/R),
# then text fields last. SemanticF1 is first (headline); token_f1 is the
# diagnostic secondary column.
_CSV_METRIC_KEYS = (
    "semantic_f1",       # headline — DSPy LLM-estimated F1
    "token_f1",          # diagnostic — SQuAD lexical F1 (no LLM)
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


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def summarize(cases: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case scores and metrics into a run summary.

    Headline: ``mean_semantic_f1`` — DSPy SemanticF1, the single honest metric.
    Diagnostic: ``mean_token_f1`` — lexical token-F1.
    Per-category SemanticF1 breakdown is included for systematic review.
    """
    # SemanticF1 — pulled from metrics dict (populated by the pipeline scorer).
    sf1_vals = [
        float(case.metrics.get("semantic_f1", case.score))
        for case in cases
    ]
    tok_f1_vals = [
        float(case.metrics.get("token_f1", 0.0))
        for case in cases
    ]

    # Per-category SemanticF1 breakdown.
    sf1_by_cat: dict[str, list[float]] = {}
    tok_f1_by_cat: dict[str, list[float]] = {}
    for case in cases:
        cat = case.category or "uncategorized"
        sf1_by_cat.setdefault(cat, []).append(
            float(case.metrics.get("semantic_f1", case.score))
        )
        tok_f1_by_cat.setdefault(cat, []).append(
            float(case.metrics.get("token_f1", 0.0))
        )

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
        # Headline
        "mean_semantic_f1": round(_mean(sf1_vals), 4),
        # Diagnostic
        "mean_token_f1": round(_mean(tok_f1_vals), 4),
        # Legacy alias kept for backward compatibility with older report readers.
        "avg_score": round(_mean(sf1_vals), 4),
        # Per-category SemanticF1
        "semantic_f1_by_category": {
            cat: round(_mean(vals), 4)
            for cat, vals in sorted(sf1_by_cat.items())
        },
        # Per-category token-F1 (diagnostic)
        "token_f1_by_category": {
            cat: round(_mean(vals), 4)
            for cat, vals in sorted(tok_f1_by_cat.items())
        },
        "metric_summary": {
            key: round(_mean(vals), 4)
            for key, vals in sorted(metric_buckets.items())
            if vals
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
        "",
        "## Headline metric",
        "",
        f"**Mean SemanticF1**: `{summary.get('mean_semantic_f1', 0.0):.3f}`  "
        f"_(diagnostic token-F1: `{summary.get('mean_token_f1', 0.0):.3f}`)_",
        "",
        "## SemanticF1 by category",
        "",
    ]
    tok_cat = summary.get("token_f1_by_category", {})
    for cat, sf1 in summary.get("semantic_f1_by_category", {}).items():
        tok = tok_cat.get(cat, 0.0)
        lines.append(f"- `{cat}`: SemanticF1 `{sf1:.3f}`  |  token-F1 `{tok:.3f}`")
    lines.extend(["", "## Aggregated metrics", ""])
    for key, val in summary.get("metric_summary", {}).items():
        lines.append(f"- `{key}`: `{val:.4f}`")
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
        sf1 = metrics.get("semantic_f1", case.get("score", 0.0))
        tok = metrics.get("token_f1", 0.0)
        lines.append(f"- SemanticF1: `{sf1:.3f}`  token-F1: `{tok:.3f}`")
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


def compare_modes(
    llm_free_cases: list[CaseResult],
    llm_assisted_cases: list[CaseResult],
) -> str:
    """Return a side-by-side per-category SemanticF1 comparison table (Markdown).

    Used by ``--compare`` in the CLI to produce the systematic-review surface.
    """
    def _cat_sf1(cases: list[CaseResult]) -> dict[str, float]:
        by_cat: dict[str, list[float]] = {}
        for case in cases:
            cat = case.category or "uncategorized"
            by_cat.setdefault(cat, []).append(
                float(case.metrics.get("semantic_f1", case.score))
            )
        return {cat: round(_mean(vals), 4) for cat, vals in sorted(by_cat.items())}

    free_sf1 = _cat_sf1(llm_free_cases)
    asst_sf1 = _cat_sf1(llm_assisted_cases)
    all_cats = sorted(set(free_sf1) | set(asst_sf1))

    free_overall = round(_mean([float(c.metrics.get("semantic_f1", c.score)) for c in llm_free_cases]), 4)
    asst_overall = round(_mean([float(c.metrics.get("semantic_f1", c.score)) for c in llm_assisted_cases]), 4)

    lines = [
        "# SemanticF1 Mode Comparison",
        "",
        f"| Category | llm_free | llm_assisted | Δ (assisted − free) |",
        f"|----------|----------|--------------|---------------------|",
    ]
    for cat in all_cats:
        f = free_sf1.get(cat, 0.0)
        a = asst_sf1.get(cat, 0.0)
        delta = a - f
        sign = "+" if delta >= 0 else ""
        lines.append(f"| `{cat}` | {f:.3f} | {a:.3f} | {sign}{delta:.3f} |")

    # Totals row
    delta_overall = asst_overall - free_overall
    sign_overall = "+" if delta_overall >= 0 else ""
    lines.append(f"| **Overall** | **{free_overall:.3f}** | **{asst_overall:.3f}** | **{sign_overall}{delta_overall:.3f}** |")
    lines.append("")
    lines.append(f"_llm_free n={len(llm_free_cases)}  llm_assisted n={len(llm_assisted_cases)}_")
    return "\n".join(lines) + "\n"
