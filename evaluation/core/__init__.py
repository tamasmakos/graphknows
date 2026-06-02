"""Shared evaluation infrastructure for graphknows benchmark runners."""

from .client import GraphknowsMCPClient, MCPClientError, normalize_tool_result
from .metrics import (
    ExpectedEvidence,
    ToolObservation,
    aggregate_observations,
    observation_to_record,
    score_observation,
    validate_result_shape,
)
from .reporting import (
    BenchmarkReportPaths,
    ReportPaths,
    render_markdown,
    write_benchmark_report,
    write_report,
)
from .stm_inspection import derive_inspect_command, inspect_session

__all__ = [
    "BenchmarkReportPaths",
    "ExpectedEvidence",
    "GraphknowsMCPClient",
    "MCPClientError",
    "ReportPaths",
    "ToolObservation",
    "aggregate_observations",
    "normalize_tool_result",
    "observation_to_record",
    "render_markdown",
    "score_observation",
    "derive_inspect_command",
    "inspect_session",
    "validate_result_shape",
    "write_benchmark_report",
    "write_report",
]
