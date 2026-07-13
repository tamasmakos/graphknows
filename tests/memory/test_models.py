"""Smoke tests for memory DTOs."""

from __future__ import annotations

import pytest

from graphknows.ingestion.models import PromotionRecord, PromotionRequest
from graphknows.models import EvidenceItem, IngestRequest, IngestResult, MemoryClass


def test_ingest_request_frozen():
    req = IngestRequest(text="hello", session_id="s1")
    with pytest.raises(Exception):
        req.text = "changed"  # type: ignore


def test_ingest_result_defaults():
    r = IngestResult(session_id="s1", file_id="f1", title="t")
    assert r.errors == []
    assert r.chunk_count == 0


def test_promotion_request_frozen():
    req = PromotionRequest(session_id="s1")
    with pytest.raises(Exception):
        req.run_gds = True  # type: ignore


def test_promotion_record_errors_default():
    r = PromotionRecord(session_id="s1")
    assert r.errors == []


def test_evidence_item_memory_class():
    # Use a real MemoryClass value
    mc = next(iter(MemoryClass))
    item = EvidenceItem(artifact_id="a1", memory_class=mc)
    assert item.memory_class == mc
