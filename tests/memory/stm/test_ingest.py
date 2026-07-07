"""Unit tests for _STMIngestHandler in isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from graphknows.ingestion.parsers.base import ParsedFile, TextSegment
from graphknows.memory.extractor import ExtractionResult
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.stm.ingest import _STMIngestHandler

import pytest

pytestmark = pytest.mark.asyncio


def _make_parsed(n_segments: int = 2) -> ParsedFile:
    return ParsedFile(
        source_id="abc123",
        title="Test Doc",
        parser_name="PlainTextParser",
        segments=[
            TextSegment(
                text=f"Section {i} content.",
                heading_path=f"## Section {i}",
                char_offset=i * 100,
            )
            for i in range(n_segments)
        ],
    )


def _make_handler(extractor=None, embed_fn=None):
    mg = MagicMock()
    pg = MagicMock()
    for attr in [
        "write_session_metadata",
        "write_file_node",
        "write_chunk_node",
        "write_entity_node",
        "write_chunk_entity_edge",
        "write_chunk_entities_bulk",
        "write_relation_edge",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    pg.write_stm_chunk = AsyncMock(return_value=None)
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    handler = _STMIngestHandler(mg, pg, extractor=extractor, embed_fn=embed)
    return handler, mg, pg, embed


async def test_ingest_parsed_returns_ingest_result():
    handler, mg, pg, embed = _make_handler()
    parsed = _make_parsed(2)
    result = await handler.ingest_parsed(parsed, "session1")
    assert isinstance(result, IngestResult)
    assert result.session_id == "session1"
    assert result.chunk_count == 2


async def test_ingest_text_no_extractor():
    handler, mg, pg, embed = _make_handler(extractor=None)
    result = await handler.ingest_text("Hello world", session_id="s1", title="Test")
    assert result.entity_count == 0
    assert result.chunk_count >= 1


async def test_ingest_text_with_extractor():
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[{"name": "Alice", "type": "PERSON", "score": 0.9}]
    )
    handler, mg, pg, embed = _make_handler(extractor=extractor)
    result = await handler.ingest_text("Alice is here", session_id="s2")
    assert result.entity_count == 1


async def test_write_session_metadata_called():
    handler, mg, pg, embed = _make_handler()
    await handler.ingest_text("x", session_id="s3")
    mg.write_session_metadata.assert_called_once()


async def test_write_stm_chunk_called_per_segment():
    handler, mg, pg, embed = _make_handler()
    parsed = _make_parsed(3)
    await handler.ingest_parsed(parsed, "s4")
    assert pg.write_stm_chunk.call_count == 3


async def test_rel_extractor_skipped_with_fewer_than_2_entities():
    rel_extractor = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[{"name": "Alice", "type": "PERSON", "score": 0.9}]
    )
    handler, mg, pg, embed = _make_handler(extractor=extractor)
    handler._rel_extractor = rel_extractor
    await handler.ingest_text("Alice alone", session_id="s5")
    rel_extractor.extract.assert_not_called()
