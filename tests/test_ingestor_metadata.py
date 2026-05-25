"""Tests for chunk metadata persistence in stm_chunks."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from graphknows.memory.extractor import ExtractionResult
from graphknows.memory.ingestor import MemoryIngestor


def test_entity_names_stored_in_chunk_metadata():
    """entity_names extracted by GLiNER2 must be persisted in stm_chunks.metadata."""
    mg = MagicMock()
    pg = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(entities=[
        {"name": "GraphKnows", "type": "ORGANIZATION", "score": 0.9},
        {"name": "Neo4j", "type": "PRODUCT", "score": 0.8},
    ])
    embed_fn = MagicMock(return_value=[0.1] * 1536)

    # Patch _run to execute synchronously
    with patch("graphknows.memory.ingestor._run", side_effect=lambda coro: None):
        ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=embed_fn)
        ingestor.ingest_text("GraphKnows uses Neo4j for storage.", session_id="test-meta")

    # Find the write_stm_chunk call
    write_calls = [c for c in pg.method_calls if "write_stm_chunk" in str(c)]
    assert write_calls, "write_stm_chunk must be called"
    chunk_dict = write_calls[0].args[0]
    meta = chunk_dict.get("metadata", {})
    assert "entity_names" in meta, "entity_names must be in chunk metadata"
    assert "GraphKnows" in meta["entity_names"]
    assert "Neo4j" in meta["entity_names"]
    assert meta.get("atom_type") == "text"
    assert meta.get("memory_class") == "raw_evidence"


def test_entity_names_from_metadata_in_recall_result():
    """RecallResult.entity_names must be populated from stm_chunks.metadata."""
    from graphknows.retrieval.recall_stm import _run_stm_ann, RecallResult

    pg = MagicMock()
    chunk = {
        "id": "chunk-001",
        "text": "GraphKnows uses Neo4j.",
        "heading_path": "",
        "file_id": "file-001",
        "cosine_score": 0.85,
        "metadata": {
            "entity_names": ["GraphKnows", "Neo4j"],
            "memory_class": "fact",
        },
    }
    with patch("graphknows.retrieval.recall_stm._run", return_value=[chunk]):
        results = {}
        _run_stm_ann(pg, "test-meta", [0.1] * 1536, 5, results)

    assert "chunk-001" in results
    _, recall = results["chunk-001"]
    assert recall.entity_names == ["GraphKnows", "Neo4j"]
