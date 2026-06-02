"""Report writers for graphknows evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.core.metrics import ToolObservation, aggregate_observations, observation_to_record


@dataclass(frozen=True)
class ReportPaths:
    """Paths written by an MCP evaluation run."""

    output_dir: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class BenchmarkReportPaths:
    """Paths written by a benchmark evaluation run."""

    output_dir: Path
    json_path: Path
    markdown_path: Path


def _format_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Graphknows MCP Evaluation Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Total calls: `{report['summary']['total_calls']}`",
        f"- Success rate: `{report['summary']['success_rate']:.3f}`",
        f"- Error rate: `{report['summary']['error_rate']:.3f}`",
        f"- Throughput: `{report['summary']['throughput_calls_per_s']:.3f}` calls/s",
        "",
        "## Tool Summary",
        "",
        "| Tool | Calls | Success | Returned Errors | p50 ms | p95 ms | Recall@k | Precision@k | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]["tools"]:
        lines.append(
            "| {tool_name} | {calls} | {success_rate:.3f} | {returned_errors} | "
            "{p50} | {p95} | {recall} | {precision} | {support} |".format(
                tool_name=row["tool_name"],
                calls=row["calls"],
                success_rate=row["success_rate"],
                returned_errors=row["returned_errors"],
                p50=_format_optional(row["p50_latency_ms"]),
                p95=_format_optional(row["p95_latency_ms"]),
                recall=_format_optional(row["avg_recall_at_k"]),
                precision=_format_optional(row["avg_precision_at_k"]),
                support=_format_optional(row["answer_support_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def _metric_summary(cases: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for case in cases:
        metrics = case.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool):
                buckets.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in sorted(buckets.items()) if values}


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Graphknows Benchmark Report: {report['benchmark']}",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Cases: `{summary['case_count']}`",
    ]
    if summary.get("scored_case_count", 0):
        lines.append(f"- Average score: `{summary['avg_score']:.3f}`")
    lines.append("")
    lines.append("## Aggregated Metrics")
    lines.append("")
    if summary.get("metric_summary"):
        for key, value in summary["metric_summary"].items():
            lines.append(f"- `{key}`: `{value:.3f}`")
    else:
        lines.append("- No aggregated metrics recorded.")
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        lines.append(f"### {case.get('case_id', 'case')}")
        lines.append("")
        lines.append(str(case.get("question", "")))
        lines.append("")
        if case.get("score") is not None:
            lines.append(f"- Score: `{case['score']:.3f}`")
        if case.get("metrics"):
            for key, value in case["metrics"].items():
                lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    observations: list[ToolObservation],
    output_dir: str | Path,
    run_id: str,
    top_k: int = 5,
    metadata: dict[str, Any] | None = None,
) -> ReportPaths:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
        "summary": aggregate_observations(observations, top_k=top_k),
        "observations": [observation_to_record(obs, top_k=top_k) for obs in observations],
    }
    json_path = out / f"{run_id}.json"
    markdown_path = out / f"{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return ReportPaths(output_dir=out, json_path=json_path, markdown_path=markdown_path)


def write_benchmark_report(
    *,
    benchmark: str,
    run_id: str,
    output_dir: str | Path,
    cases: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> BenchmarkReportPaths:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = [float(case["score"]) for case in cases if isinstance(case.get("score"), (int, float))]
    report = {
        "benchmark": benchmark,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "case_count": len(cases),
            "scored_case_count": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "metric_summary": _metric_summary(cases),
        },
        "cases": cases,
    }
    json_path = out / f"{run_id}-{benchmark}.json"
    markdown_path = out / f"{run_id}-{benchmark}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_benchmark_markdown(report), encoding="utf-8")
    return BenchmarkReportPaths(output_dir=out, json_path=json_path, markdown_path=markdown_path)
