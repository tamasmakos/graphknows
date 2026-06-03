"""Shared evaluation infrastructure for graphknows benchmark runners."""

from .client import GraphknowsMCPClient, MCPClientError, normalize_tool_result
from .models import BenchmarkCaseResult
from .reporting import (
    BenchmarkReportPaths,
    ExpectedEvidence,
    ReportPaths,
    ToolObservation,
    aggregate_observations,
    observation_to_record,
    render_benchmark_markdown,
    render_markdown,
    score_observation,
    validate_result_shape,
    write_benchmark_report,
    write_report,
)

__all__ = [
    "BenchmarkCaseResult",
    "BenchmarkReportPaths",
    "ExpectedEvidence",
    "GraphknowsMCPClient",
    "MCPClientError",
    "ReportPaths",
    "ToolObservation",
    "aggregate_observations",
    "normalize_tool_result",
    "observation_to_record",
    "render_benchmark_markdown",
    "render_markdown",
    "score_observation",
    "validate_result_shape",
    "write_benchmark_report",
    "write_report",
]
