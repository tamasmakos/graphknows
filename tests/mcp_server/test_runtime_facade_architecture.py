"""Source-level guardrails for the MCP runtime facade."""

from __future__ import annotations

from pathlib import Path


def test_runtime_facade_exists_for_transport_neutral_operations() -> None:
    """The runtime facade is the transport-neutral seam for memory operations."""
    text = Path("graphknows/runtime.py").read_text(encoding="utf-8")
    assert "class Memory" in text
    assert "def recall_memory" in text
    assert "def corpus_ingest" in text


def test_corpus_workflow_does_not_depend_on_mcp_state() -> None:
    """Memory workflows should not depend inward on MCP transport state."""
    text = Path("graphknows/memory/workflows/corpus_ingest.py").read_text(encoding="utf-8")
    assert "graphknows.mcp._state" not in text
