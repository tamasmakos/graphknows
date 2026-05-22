"""Unit tests for MemoryIngestor."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.ingestion.parsers.base import ParsedFile, TextSegment
from graphknows.memory.ingestor import IngestResult, MemoryIngestor


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
        "write_session_metadata", "write_file_node", "write_chunk_node",
        "write_entity_node", "write_chunk_entity_edge",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    pg.write_stm_chunk = AsyncMock(return_value=None)
    extractor = extractor or MagicMock(return_value=[{"name": "EU", "type": "ORG", "score": 0.9}])
    # extractor.extract() not extractor()
    ext_obj = MagicMock()
    ext_obj.extract = extractor
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    return MemoryIngestor(mg, pg, ext_obj, embed_fn=embed), mg, pg, ext_obj, embed


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
        return_value=[
            {"name": "EU", "type": "ORG", "score": 0.9},
            {"name": "Germany", "type": "LOC", "score": 0.8},
        ]
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
