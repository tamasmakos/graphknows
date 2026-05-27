"""Tests for graphknows.graphgen.ingest."""

from __future__ import annotations


def test_run_ingestion_is_callable() -> None:
    """run_ingestion must exist and be callable."""
    from graphknows.graphgen.ingest import run_ingestion

    assert callable(run_ingestion)
