"""Tests for graphknows.graphgen.ingest."""

from __future__ import annotations

import inspect


def test_run_ingestion_is_callable() -> None:
    """run_ingestion must exist and be callable."""
    from graphknows.graphgen.ingest import run_ingestion

    assert callable(run_ingestion)


def test_run_ingestion_is_coroutine_function() -> None:
    """run_ingestion must be an async function (coroutine)."""
    from graphknows.graphgen.ingest import run_ingestion

    assert inspect.iscoroutinefunction(run_ingestion), (
        "run_ingestion must be async so callers can await it"
    )


def test_run_ingestion_signature() -> None:
    """run_ingestion must accept a documents parameter."""
    from graphknows.graphgen.ingest import run_ingestion

    sig = inspect.signature(run_ingestion)
    params = set(sig.parameters)
    assert "documents" in params, "run_ingestion missing required param: documents"
