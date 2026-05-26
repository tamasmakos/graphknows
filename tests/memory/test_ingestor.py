"""Unit tests for MemoryIngestor — core ingestion, metadata persistence, and rel_extractor wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.ingestion.parsers.base import ParsedFile, TextSegment
from graphknows.memory.extractor import ExtractionResult
from graphknows.memory.ingestor import IngestResult, MemoryIngestor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_parsed(n_segments: int = 2) -> ParsedFile:
    segments = [
        TextSegment(
            text=f"## Section {i}\n\nContent of section {i}.",
            heading_path=f"## Section {i}",
            char_offset=i * 100,
        )
        for i in range(n_segments)
    ]
    return ParsedFile(
        source_id="abc123",
        title="Test Doc",
        parser_name="PlainTextParser",
        segments=segments,
    )


def _make_ingestor(extractor=None, embed_fn=None):
    mg = MagicMock()
    pg = MagicMock()
    # All store methods are async — mock as coroutines so _run() works
    for attr in [
        "write_session_metadata",
        "write_file_node",
        "write_chunk_node",
        "write_entity_node",
        "write_chunk_entity_edge",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    pg.write_stm_chunk = AsyncMock(return_value=None)
    extractor = extractor or MagicMock(
        return_value=ExtractionResult(entities=[{"name": "EU", "type": "ORG", "score": 0.9}])
    )
    # extractor.extract() not extractor()
    ext_obj = MagicMock()
    ext_obj.extract = extractor
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    return MemoryIngestor(mg, pg, ext_obj, embed_fn=embed), mg, pg, ext_obj, embed


# ---------------------------------------------------------------------------
# Core ingestor tests (test_memory_ingestor.py)
# ---------------------------------------------------------------------------


def test_ingest_text_calls_parser_and_ingest_parsed():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    result = ingestor.ingest_text("Hello world", session_id="s1", title="Test")
    assert isinstance(result, IngestResult)
    assert result.session_id == "s1"
    assert result.chunk_count >= 1


def test_ingest_file_reads_file_and_selects_parser(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("# Hello\n\nSome content here.")
    ingestor, mg, pg, ext, embed = _make_ingestor()
    result = ingestor.ingest_file(str(f), session_id="s1")
    assert result.chunk_count >= 1
    assert result.title == "doc.txt"


def test_ingest_parsed_calls_extractor_per_segment():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    parsed = _make_parsed(3)
    ingestor.ingest_parsed(parsed, session_id="s1")
    assert ext.extract.call_count == 3


def test_ingest_parsed_calls_embed_per_segment():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    parsed = _make_parsed(2)
    ingestor.ingest_parsed(parsed, session_id="s1")
    assert embed.call_count == 2


def test_ingest_parsed_calls_write_chunk_node_per_segment():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    parsed = _make_parsed(2)
    ingestor.ingest_parsed(parsed, session_id="s1")
    assert mg.write_chunk_node.call_count == 2


def test_ingest_parsed_calls_write_entity_node_per_entity():
    # extractor returns 2 entities per chunk, 2 chunks → 4 total
    ext_fn = MagicMock(
        return_value=ExtractionResult(
            entities=[
                {"name": "EU", "type": "ORG", "score": 0.9},
                {"name": "Germany", "type": "LOC", "score": 0.8},
            ]
        )
    )
    ingestor, mg, pg, _ext, embed = _make_ingestor(extractor=ext_fn)
    parsed = _make_parsed(2)
    result = ingestor.ingest_parsed(parsed, session_id="s1")
    assert result.entity_count == 4
    assert mg.write_entity_node.call_count == 4
    # Verify entity_type is passed through to the store call
    _, kwargs = mg.write_entity_node.call_args
    assert "entity_type" in kwargs


def test_ingest_parsed_calls_write_stm_chunk_per_segment():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    parsed = _make_parsed(2)
    ingestor.ingest_parsed(parsed, session_id="s1")
    assert pg.write_stm_chunk.call_count == 2


def test_extractor_failure_propagates():
    """Extraction errors now propagate immediately — no silent swallow."""
    ext_fn = MagicMock(side_effect=RuntimeError("NER crashed"))
    ingestor, mg, pg, _ext, embed = _make_ingestor(extractor=ext_fn)
    parsed = _make_parsed(2)
    with pytest.raises(RuntimeError, match="NER crashed"):
        ingestor.ingest_parsed(parsed, session_id="s1")


def test_embed_failure_propagates():
    """Embed errors now propagate immediately — no silent zero-vector fallback."""
    embed_fn = MagicMock(side_effect=RuntimeError("embed failed"))
    ingestor, mg, pg, ext, _embed = _make_ingestor(embed_fn=embed_fn)
    parsed = _make_parsed(1)
    with pytest.raises(RuntimeError, match="embed failed"):
        ingestor.ingest_parsed(parsed, session_id="s1")


def test_ingest_result_has_correct_counts():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    parsed = _make_parsed(3)
    result = ingestor.ingest_parsed(parsed, session_id="s2")
    assert result.chunk_count == 3
    assert result.session_id == "s2"
    assert result.file_id == parsed.source_id
    assert result.title == parsed.title


def test_ingest_empty_text_returns_no_errors():
    ingestor, mg, pg, ext, embed = _make_ingestor()
    result = ingestor.ingest_text("", session_id="s1")
    assert result.session_id == "s1"
    # PlainTextParser may produce 1 empty-content chunk — that is acceptable
    assert isinstance(result.chunk_count, int)


# ---------------------------------------------------------------------------
# Metadata persistence tests (test_ingestor_metadata.py)
# ---------------------------------------------------------------------------


def test_entity_names_stored_in_chunk_metadata():
    """entity_names extracted by GLiNER2 must be persisted in stm_chunks.metadata."""
    mg = MagicMock()
    pg = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[
            {"name": "GraphKnows", "type": "ORGANIZATION", "score": 0.9},
            {"name": "Neo4j", "type": "PRODUCT", "score": 0.8},
        ]
    )
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
    from graphknows.retrieval.recall_stm import _run_stm_ann

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


# ---------------------------------------------------------------------------
# rel_extractor wiring tests (test_ingestor_rel_extractor.py)
# ---------------------------------------------------------------------------


def _make_parsed_rel(entities_per_chunk: int = 2, n_segments: int = 1) -> ParsedFile:
    """Build a minimal ParsedFile with deterministic content."""
    segments = [
        TextSegment(
            text=f"Segment {i} with content about entities.",
            heading_path=f"Section {i}",
            char_offset=i * 100,
        )
        for i in range(n_segments)
    ]
    return ParsedFile(
        source_id="test_doc",
        title="Test",
        parser_name="PlainTextParser",
        segments=segments,
    )


def _make_mg_mock(include_write_relation_edge: bool = True) -> MagicMock:
    mg = MagicMock()
    for attr in [
        "write_session_metadata",
        "write_file_node",
        "write_chunk_node",
        "write_entity_node",
        "write_chunk_entity_edge",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    if include_write_relation_edge:
        mg.write_relation_edge = AsyncMock(return_value=None)
    return mg


def _make_extractor_rel(n_entities: int = 2) -> MagicMock:
    """Return a mock extractor that always returns *n_entities* entities."""
    entities = [{"name": f"Entity{i}", "type": "ENTITY", "score": 0.9} for i in range(n_entities)]
    ext = MagicMock()
    ext.extract = MagicMock(return_value=ExtractionResult(entities=entities))
    return ext


def _make_rel_extractor(relations: list[dict] | None = None) -> MagicMock:
    if relations is None:
        relations = [
            {
                "subject": "Entity0",
                "predicate": "RELATES_TO",
                "object": "Entity1",
                "confidence": 0.8,
            }
        ]
    rel_ext = MagicMock()
    rel_ext.extract = MagicMock(return_value=relations)
    return rel_ext


def _make_ingestor_rel(
    extractor: MagicMock | None = None,
    rel_extractor: MagicMock | None = None,
    n_entities: int = 2,
) -> tuple[MemoryIngestor, MagicMock, MagicMock]:
    mg = _make_mg_mock()
    pg = MagicMock()
    pg.write_stm_chunk = AsyncMock(return_value=None)
    ext = extractor or _make_extractor_rel(n_entities)
    embed = MagicMock(return_value=[0.1] * 10)
    ingestor = MemoryIngestor(
        mg_store=mg,
        pg_store=pg,
        extractor=ext,
        embed_fn=embed,
        rel_extractor=rel_extractor,
    )
    return ingestor, mg, pg


def test_rel_extractor_called_and_writes_edge_per_relation():
    """With rel_extractor set and >=2 entities, write_relation_edge called once per rel."""
    rel_ext = _make_rel_extractor(
        [
            {
                "subject": "Entity0",
                "predicate": "KNOWS",
                "object": "Entity1",
                "confidence": 0.9,
            },
            {
                "subject": "Entity1",
                "predicate": "OPPOSES",
                "object": "Entity0",
                "confidence": 0.7,
            },
        ]
    )
    ingestor, mg, _ = _make_ingestor_rel(rel_extractor=rel_ext, n_entities=2)
    parsed = _make_parsed_rel(n_segments=1)

    ingestor.ingest_parsed(parsed, session_id="s1")

    assert rel_ext.extract.call_count == 1
    assert mg.write_relation_edge.call_count == 2

    # Verify first call used the right kwargs
    call_kwargs = mg.write_relation_edge.call_args_list[0][1]
    assert call_kwargs["session_id"] == "s1"
    assert call_kwargs["subject_name"] == "Entity0"
    assert call_kwargs["relation_name"] == "KNOWS"
    assert call_kwargs["object_name"] == "Entity1"
    assert call_kwargs["confidence"] == pytest.approx(0.9)


def test_rel_extractor_none_does_not_call_write_relation_edge():
    """When rel_extractor is None, write_relation_edge must never be called."""
    ingestor, mg, _ = _make_ingestor_rel(rel_extractor=None, n_entities=3)
    parsed = _make_parsed_rel(n_segments=1)

    ingestor.ingest_parsed(parsed, session_id="s2")

    mg.write_relation_edge.assert_not_called()


def test_rel_extractor_skipped_when_fewer_than_two_entities():
    """When the extractor returns <2 entities, rel_extractor.extract() not called."""
    rel_ext = _make_rel_extractor()
    ingestor, mg, _ = _make_ingestor_rel(rel_extractor=rel_ext, n_entities=1)
    parsed = _make_parsed_rel(n_segments=1)

    ingestor.ingest_parsed(parsed, session_id="s3")

    rel_ext.extract.assert_not_called()
    mg.write_relation_edge.assert_not_called()


def test_rel_extractor_exception_does_not_abort_ingest():
    """Exceptions from rel_extractor.extract() are swallowed; ingest returns normally."""
    rel_ext = MagicMock()
    rel_ext.extract = MagicMock(side_effect=RuntimeError("rel model crashed"))

    ingestor, mg, pg = _make_ingestor_rel(rel_extractor=rel_ext, n_entities=2)
    parsed = _make_parsed_rel(n_segments=1)

    # Must not raise
    result = ingestor.ingest_parsed(parsed, session_id="s4")

    assert result.session_id == "s4"
    assert result.chunk_count == 1
    # write_relation_edge was never reached
    mg.write_relation_edge.assert_not_called()
    # pgvector write still happened (ingest continued)
    pg.write_stm_chunk.assert_called_once()
