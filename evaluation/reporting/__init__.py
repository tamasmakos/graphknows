"""Reporting stage — write JSON + CSV + Markdown for a run."""

from __future__ import annotations

from evaluation.reporting.report import ReportPaths, summarize, write_report

__all__ = ["ReportPaths", "summarize", "write_report"]
