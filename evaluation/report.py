"""Report writers for graphknows MCP evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import ToolObservation, aggregate_observations, observation_to_record


@dataclass(frozen=True)
class ReportPaths:
    """Paths written by an evaluation run."""

    output_dir: Path
    json_path: Path
    markdown_path: Path


def _format_optional(value: Any) -> str:
    """Format optional report values for Markdown tables."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for an evaluation report."""
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


def write_report(
    observations: list[ToolObservation],
    output_dir: str | Path,
    run_id: str,
    top_k: int = 5,
    metadata: dict[str, Any] | None = None,
) -> ReportPaths:
    """Write JSON and Markdown reports for an evaluation run."""
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

