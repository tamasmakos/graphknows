"""Tests for graphknows.graphgen.ingest."""

from __future__ import annotations

import pytest


def test_run_ingestion_emits_deprecation_warning() -> None:
    """run_ingestion must emit a DeprecationWarning on every call."""
    from unittest.mock import AsyncMock, patch

    async def _run() -> None:
        from graphknows.graphgen.ingest import run_ingestion

        with patch(
            "graphknows.graphgen.ingest.run_ingestion.__wrapped__",
            new=AsyncMock(return_value={}),
            create=True,
        ):
            pass  # patching not needed — just call and catch warn

        with pytest.warns(DeprecationWarning, match="CorpusIngestWorkflow"):
            with patch(
                "graphknows.graphgen.orchestrator.run_pipeline",
                new=AsyncMock(return_value={}),
            ):
                await run_ingestion(documents=[{"text": "hello", "id": "t1"}])

    import asyncio

    asyncio.run(_run())
