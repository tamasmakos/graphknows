"""Deterministic tool/terminal execution summarizer.

Converts raw tool call data into structured, memory-worthy evidence records.
No LLM calls. No raw stdout passthrough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from graphknows.memory.adapter_types import MemoryClass, ToolCallRecord


class ToolCategory(str, Enum):  # noqa: UP042
    """Category classification for tool names."""

    shell = "shell"
    file_read = "file_read"
    file_write = "file_write"
    web = "web"
    quality = "quality"
    graph = "graph"
    memory = "memory"
    other = "other"


@dataclass(frozen=True)
class ToolSummary:
    """Structured summary of a single tool execution.

    Attributes:
        tool_name: Name of the tool that was called.
        category: Semantic category of the tool.
        outcome: "success" | "error" | "warning" | "partial".
        signal: First meaningful signal line (not raw stdout).
        artifact_paths: Any file paths touched or produced.
        error_class: Error type if applicable; None otherwise.
        is_durable_fact: True if the summary describes a stable fact
            worth promoting.
        memory_class: Always raw_evidence for v1.
    """

    tool_name: str
    category: ToolCategory
    outcome: str
    signal: str
    artifact_paths: list[str]
    error_class: str | None
    is_durable_fact: bool
    memory_class: MemoryClass


class ToolSummarizer:
    """Deterministic tool/terminal execution summarizer.

    Converts raw tool call data into structured, memory-worthy evidence
    records. No LLM calls. No raw stdout passthrough.
    """

    _QUALITY_TOOLS: frozenset[str] = frozenset(["pytest", "mypy", "ruff", "bandit", "coverage"])
    _SHELL_TOOLS: frozenset[str] = frozenset(["terminal", "bash", "shell", "exec", "run"])
    _FILE_READ_TOOLS: frozenset[str] = frozenset(["read_file", "cat", "view"])
    _FILE_WRITE_TOOLS: frozenset[str] = frozenset(["write_file", "patch", "apply_patch"])
    _WEB_TOOLS: frozenset[str] = frozenset(["web_search", "fetch", "browser_navigate", "webfetch"])
    _GRAPH_TOOLS: frozenset[str] = frozenset(["neo4j_query", "memgraph_query", "cypher"])
    _MEMORY_TOOLS: frozenset[str] = frozenset(["recall", "ingest", "flush", "graphknows_recall"])

    def categorize(self, tool_name: str) -> ToolCategory:
        """Return the ToolCategory for a given tool name.

        Args:
            tool_name: Name of the tool to categorize.

        Returns:
            The matching ToolCategory, or ToolCategory.other if no match.
        """
        if tool_name in self._QUALITY_TOOLS:
            return ToolCategory.quality
        if tool_name in self._SHELL_TOOLS:
            return ToolCategory.shell
        if tool_name in self._FILE_READ_TOOLS:
            return ToolCategory.file_read
        if tool_name in self._FILE_WRITE_TOOLS:
            return ToolCategory.file_write
        if tool_name in self._WEB_TOOLS:
            return ToolCategory.web
        if tool_name in self._GRAPH_TOOLS:
            return ToolCategory.graph
        if tool_name in self._MEMORY_TOOLS:
            return ToolCategory.memory
        return ToolCategory.other

    def summarize(
        self,
        tool_name: str,
        input_text: str,
        output_text: str,
        exit_code: int | None = None,
        error_text: str | None = None,
    ) -> ToolSummary:
        """Produce a structured ToolSummary from raw tool call data.

        Args:
            tool_name: Name of the tool that was called.
            input_text: Human-readable input summary (not full args dump).
            output_text: Raw output text (will be summarized, not passed
                through).
            exit_code: Shell exit code for shell tools; None otherwise.
            error_text: First error line or exception type; None on success.

        Returns:
            ToolSummary with structured fields.
        """
        category = self.categorize(tool_name)
        outcome = self._determine_outcome(
            exit_code=exit_code,
            error_text=error_text,
            output_text=output_text,
        )
        signal = self._extract_signal(
            category=category,
            input_text=input_text,
            output_text=output_text,
        )
        artifact_paths = self._extract_artifact_paths(
            input_text=input_text,
            output_text=output_text,
        )
        error_class = self._extract_error_class(
            error_text=error_text,
            exit_code=exit_code,
        )
        is_durable_fact = self._determine_durable_fact(
            category=category,
            outcome=outcome,
            output_text=output_text,
        )

        return ToolSummary(
            tool_name=tool_name,
            category=category,
            outcome=outcome,
            signal=signal,
            artifact_paths=artifact_paths,
            error_class=error_class,
            is_durable_fact=is_durable_fact,
            memory_class=MemoryClass.raw_evidence,
        )

    def _determine_outcome(
        self,
        exit_code: int | None,
        error_text: str | None,
        output_text: str,
    ) -> str:
        """Determine the execution outcome based on provided signals."""
        if exit_code is not None and exit_code != 0:
            return "error"
        if error_text is not None:
            return "error"
        if "warning" in output_text.lower()[:500]:
            return "warning"
        return "success"

    def _extract_signal(
        self,
        category: ToolCategory,
        input_text: str,
        output_text: str,
    ) -> str:
        """Extract the first meaningful signal line from tool output."""
        if category == ToolCategory.quality:
            return self._extract_quality_signal(output_text)
        if category == ToolCategory.shell:
            return self._extract_shell_signal(output_text)
        if category == ToolCategory.file_read:
            return self._extract_file_read_signal(input_text)
        if category == ToolCategory.file_write:
            return self._extract_file_write_signal(input_text)
        if category == ToolCategory.web:
            return self._extract_web_signal(output_text)
        return self._extract_default_signal(output_text)

    def _extract_quality_signal(self, output_text: str) -> str:
        """Extract the last summary line from quality tool output."""
        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        summary_keywords = [
            "passed",
            "failed",
            "error",
            "found",
            "checks",
            "summary",
        ]
        for line in reversed(lines):
            lower = line.lower()
            if any(keyword in lower for keyword in summary_keywords):
                return line[:120]
        # Fallback to last non-empty line if no keyword match.
        if lines:
            return lines[-1][:120]
        return ""

    def _extract_shell_signal(self, output_text: str) -> str:
        """Extract the first non-empty, non-path line from shell output."""
        for line in output_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[/~]\S+", stripped):
                continue
            return stripped[:120]
        return ""

    def _extract_file_read_signal(self, input_text: str) -> str:
        """Build a signal for file read tools based on input text."""
        path = self._extract_first_path(input_text)
        if path:
            return f"Read {path}"
        return "Read file"

    def _extract_file_write_signal(self, input_text: str) -> str:
        """Build a signal for file write tools based on input text."""
        path = self._extract_first_path(input_text)
        if path:
            return f"Wrote {path}"
        return "Wrote file"

    def _extract_web_signal(self, output_text: str) -> str:
        """Extract the first sentence from web tool output."""
        text = output_text.strip()
        if not text:
            return ""
        match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        if match:
            return match[0][:120]
        return text[:120]

    def _extract_default_signal(self, output_text: str) -> str:
        """Extract first 120 chars of output text stripped of whitespace."""
        stripped = output_text.strip()
        return stripped[:120] if stripped else ""

    def _extract_first_path(self, text: str) -> str | None:
        """Extract the first path-like string from text."""
        match = re.search(r"[/~]\S+\.\w+", text)
        return match.group(0) if match else None

    def _extract_artifact_paths(
        self,
        input_text: str,
        output_text: str,
    ) -> list[str]:
        """Extract file paths from input and output text.

        Returns:
            Deduplicated list of up to 10 artifact paths.
        """
        combined = f"{input_text}\n{output_text}"
        paths = re.findall(r"[/~]\S+\.\w+", combined)
        seen: set[str] = set()
        deduped: list[str] = []
        for path in paths:
            if path not in seen and len(deduped) < 10:
                seen.add(path)
                deduped.append(path)
        return deduped

    def _extract_error_class(
        self,
        error_text: str | None,
        exit_code: int | None,
    ) -> str | None:
        """Extract the error class from error text or exit code."""
        if error_text is not None:
            match = re.search(r"\b(\w+(?:Error|Exception))", error_text)
            if match:
                return match.group(1)
        if exit_code is not None and exit_code != 0:
            return "NonZeroExit"
        return None

    def _determine_durable_fact(
        self,
        category: ToolCategory,
        outcome: str,
        output_text: str,
    ) -> bool:
        """Determine whether this summary represents a durable fact."""
        if category == ToolCategory.quality and outcome == "success":
            return True
        if category == ToolCategory.file_write and outcome == "success":
            return True
        return bool(
            category == ToolCategory.graph
            and outcome == "success"
            and ("nodes" in output_text.lower() or "relationships" in output_text.lower())
        )


def summarize_tool_call(record: ToolCallRecord) -> ToolSummary:
    """Convenience wrapper to summarize a ToolCallRecord.

    Args:
        record: The tool call record to summarize.

    Returns:
        A structured ToolSummary.
    """
    summarizer = ToolSummarizer()
    return summarizer.summarize(
        tool_name=record.tool_name,
        input_text=record.input_summary,
        output_text=record.output_summary,
        exit_code=record.exit_code,
        error_text=record.error,
    )
