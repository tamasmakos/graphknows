"""Tests for rel_extractor wiring in MemoryIngestor._ingest_parsed()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.parsers.base import ParsedFile, TextSegment
from graphknows.memory.ingestor import MemoryIngestor


def _make_parsed(entities_per_chunk: int = 2, n_segments: int = 1) -> ParsedFile:
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


def _make_extractor(n_entities: int = 2) -> MagicMock:
    """Return a mock extractor that always returns *n_entities* entities."""
    entities = [{"name": f"Entity{i}", "type": "ENTITY", "score": 0.9} for i in range(n_entities)]
    ext = MagicMock()
    ext.extract = MagicMock(return_value=entities)
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


def _make_ingestor(
    extractor: MagicMock | None = None,
    rel_extractor: MagicMock | None = None,
    n_entities: int = 2,
) -> tuple[MemoryIngestor, MagicMock, MagicMock]:
    mg = _make_mg_mock()
    pg = MagicMock()
    pg.write_stm_chunk = AsyncMock(return_value=None)
    ext = extractor or _make_extractor(n_entities)
    embed = MagicMock(return_value=[0.1] * 10)
    ingestor = MemoryIngestor(
        mg_store=mg,
        pg_store=pg,
        extractor=ext,
        embed_fn=embed,
        rel_extractor=rel_extractor,
    )
    return ingestor, mg, pg


# ---------------------------------------------------------------------------
# Test 1: rel_extractor is called and write_relation_edge fires per relation
# ---------------------------------------------------------------------------


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
    ingestor, mg, _ = _make_ingestor(rel_extractor=rel_ext, n_entities=2)
    parsed = _make_parsed(n_segments=1)

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


# ---------------------------------------------------------------------------
# Test 2: rel_extractor is None → write_relation_edge is never called
# ---------------------------------------------------------------------------


def test_rel_extractor_none_does_not_call_write_relation_edge():
    """When rel_extractor is None, write_relation_edge must never be called."""
    ingestor, mg, _ = _make_ingestor(rel_extractor=None, n_entities=3)
    parsed = _make_parsed(n_segments=1)

    ingestor.ingest_parsed(parsed, session_id="s2")

    mg.write_relation_edge.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: len(entities) < 2 → write_relation_edge is never called
# ---------------------------------------------------------------------------


def test_rel_extractor_skipped_when_fewer_than_two_entities():
    """When the extractor returns <2 entities, rel_extractor.extract() not called."""
    rel_ext = _make_rel_extractor()
    ingestor, mg, _ = _make_ingestor(rel_extractor=rel_ext, n_entities=1)
    parsed = _make_parsed(n_segments=1)

    ingestor.ingest_parsed(parsed, session_id="s3")

    rel_ext.extract.assert_not_called()
    mg.write_relation_edge.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: rel_extractor.extract() raises → ingest continues, no propagation
# ---------------------------------------------------------------------------


def test_rel_extractor_exception_does_not_abort_ingest():
    """Exceptions from rel_extractor.extract() are swallowed; ingest returns normally."""
    rel_ext = MagicMock()
    rel_ext.extract = MagicMock(side_effect=RuntimeError("rel model crashed"))

    ingestor, mg, pg = _make_ingestor(rel_extractor=rel_ext, n_entities=2)
    parsed = _make_parsed(n_segments=1)

    # Must not raise
    result = ingestor.ingest_parsed(parsed, session_id="s4")

    assert result.session_id == "s4"
    assert result.chunk_count == 1
    # write_relation_edge was never reached
    mg.write_relation_edge.assert_not_called()
    # pgvector write still happened (ingest continued)
    pg.write_stm_chunk.assert_called_once()
