"""Tests for the deterministic tool/terminal execution summarizer."""

from __future__ import annotations

import pytest

from graphknows.memory.adapter_types import MemoryClass, ToolCallRecord
from graphknows.memory.tool_summarizer import (
    ToolCategory,
    ToolSummarizer,
    ToolSummary,
    summarize_tool_call,
)


class TestCategorize:
    """Tests for ToolSummarizer.categorize."""

    def test_categorize_pytest(self) -> None:
        summarizer = ToolSummarizer()
        result = summarizer.categorize("pytest")
        assert result == ToolCategory.quality

    def test_categorize_terminal(self) -> None:
        summarizer = ToolSummarizer()
        result = summarizer.categorize("terminal")
        assert result == ToolCategory.shell

    def test_categorize_read_file(self) -> None:
        summarizer = ToolSummarizer()
        result = summarizer.categorize("read_file")
        assert result == ToolCategory.file_read

    def test_categorize_unknown(self) -> None:
        summarizer = ToolSummarizer()
        result = summarizer.categorize("my_custom_tool")
        assert result == ToolCategory.other


class TestSummarize:
    """Tests for ToolSummarizer.summarize."""

    def test_summarize_pytest_success(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="pytest",
            input_text="pytest tests/",
            output_text="test_foo.py::test_example PASSED\n\n27 passed in 0.17s\n",
            exit_code=0,
        )
        assert summary.outcome == "success"
        assert "27 passed" in summary.signal
        assert summary.is_durable_fact is True

    def test_summarize_pytest_failure(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="pytest",
            input_text="pytest tests/",
            output_text="test_foo.py::test_example FAILED\n\n3 failed in 0.5s\n",
            exit_code=1,
        )
        assert summary.outcome == "error"
        assert "3 failed" in summary.signal
        assert summary.is_durable_fact is False

    def test_summarize_terminal_success(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="terminal",
            input_text="ls -la",
            output_text="total 128\ndrwxr-xr-x  5 user user  4096 Jan  1 00:00 .\n",
            exit_code=0,
        )
        assert summary.outcome == "success"

    def test_summarize_terminal_error(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="terminal",
            input_text="false",
            output_text="",
            exit_code=1,
            error_text="CalledProcessError",
        )
        assert summary.outcome == "error"
        assert summary.error_class == "CalledProcessError"

    def test_summarize_file_write_success(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="write_file",
            input_text="write_file /tmp/test.py",
            output_text="File written successfully.",
            exit_code=0,
        )
        assert "/tmp/test.py" in summary.artifact_paths
        assert summary.is_durable_fact is True

    def test_summarize_traceback(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="terminal",
            input_text="python script.py",
            output_text="Traceback (most recent call last):\n  ...",
            exit_code=1,
            error_text="ImportError: No module named X",
        )
        assert summary.error_class == "ImportError"

    def test_summarize_nonzero_exit_no_error_text(self) -> None:
        summarizer = ToolSummarizer()
        summary = summarizer.summarize(
            tool_name="terminal",
            input_text="some_command",
            output_text="",
            exit_code=2,
        )
        assert summary.error_class == "NonZeroExit"


class TestSummarizeToolCall:
    """Tests for the summarize_tool_call convenience function."""

    def test_summarize_tool_call_convenience_fn(self) -> None:
        record = ToolCallRecord(
            tool_id="call-1",
            tool_name="pytest",
            input_summary="pytest tests/",
            output_summary="27 passed in 0.17s",
            exit_code=0,
        )
        summary = summarize_tool_call(record)
        assert isinstance(summary, ToolSummary)
        assert summary.tool_name == "pytest"
        assert summary.outcome == "success"


class TestToolSummary:
    """Tests for ToolSummary dataclass behavior."""

    def test_summary_is_frozen(self) -> None:
        summary = ToolSummary(
            tool_name="pytest",
            category=ToolCategory.quality,
            outcome="success",
            signal="27 passed",
            artifact_paths=[],
            error_class=None,
            is_durable_fact=True,
            memory_class=MemoryClass.raw_evidence,
        )
        with pytest.raises(
            AttributeError,
            match="cannot assign to field|frozen instance",
        ):
            summary.outcome = "error"  # type: ignore[misc]

    def test_no_raw_stdout_in_signal(self) -> None:
        summarizer = ToolSummarizer()
        long_output = "x" * 1000
        summary = summarizer.summarize(
            tool_name="terminal",
            input_text="echo stuff",
            output_text=long_output,
            exit_code=0,
        )
        assert len(summary.signal) <= 120
