"""Unit tests for _STMIngestHandler in isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.parsers.text import ParsedFile, TextSegment
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.models.ingest import IngestResult

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
    store = MagicMock()
    for attr in [
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_relation",
        "write_temporal",
    ]:
        setattr(store, attr, AsyncMock(return_value=None))
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    handler = _STMIngestHandler(store, extractor=extractor, embed_fn=embed)
    return handler, store, embed


async def test_ingest_parsed_returns_ingest_result():
    handler, store, embed = _make_handler()
    parsed = _make_parsed(2)
    result = await handler.ingest_parsed(parsed, "session1")
    assert isinstance(result, IngestResult)
    assert result.session_id == "session1"
    assert result.chunk_count == 2


async def test_ingest_text_no_extractor():
    handler, store, embed = _make_handler(extractor=None)
    result = await handler.ingest_text("Hello world", session_id="s1", title="Test")
    assert result.entity_count == 0
    assert result.chunk_count >= 1


async def test_ingest_text_with_extractor():
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[{"name": "Alice", "type": "PERSON", "score": 0.9}]
    )
    handler, store, embed = _make_handler(extractor=extractor)
    result = await handler.ingest_text("Alice is here", session_id="s2")
    assert result.entity_count == 1


async def test_ensure_session_called():
    handler, store, embed = _make_handler()
    await handler.ingest_text("x", session_id="s3")
    store.ensure_session.assert_awaited_once()


async def test_write_chunk_called_per_segment():
    handler, store, embed = _make_handler()
    parsed = _make_parsed(3)
    await handler.ingest_parsed(parsed, "s4")
    assert store.write_chunk.await_count == 3


async def test_rel_extractor_skipped_with_fewer_than_2_entities():
    rel_extractor = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[{"name": "Alice", "type": "PERSON", "score": 0.9}]
    )
    handler, store, embed = _make_handler(extractor=extractor)
    handler._rel_extractor = rel_extractor
    await handler.ingest_text("Alice alone", session_id="s5")
    rel_extractor.extract.assert_not_called()


async def test_ingest_turn_writes_the_timestamp_onto_the_chunk():
    """CHUNK.ts must carry the turn's timestamp, or retrieval returns undated text.

    Measured 27 Jul 2026: NOTHING ever populated CHUNK.ts. The column was
    declared, written, indexed and selected by both read queries, and was ""
    for every chunk in every namespace on disk — the storage layer looked
    correct end to end while carrying no data. The timestamp was already in
    hand here as ``anchor`` (it anchors relative-date resolution); it simply
    never rode into the chunk dict. Asserts on the VALUE.
    """
    handler, store, _ = _make_handler(extractor=None)

    await handler.ingest_turn(
        "s1",
        "f1",
        0,
        "user",
        "Mochi's vet is Dr. Nagy.",
        metadata={"timestamp": "4:04 pm on 20 January, 2023"},
        speaker="Gina",
    )

    chunk = store.write_chunk.await_args.args[0]
    assert chunk["ts"] == "4:04 pm on 20 January, 2023"
    assert chunk["speaker"] == "Gina"


async def test_ingest_turn_without_a_timestamp_writes_empty_not_ingest_time():
    """No timestamp means no timestamp — never "now", which would date the memory wrong."""
    handler, store, _ = _make_handler(extractor=None)

    await handler.ingest_turn("s1", "f1", 0, "user", "Hello there.", speaker="Gina")

    assert store.write_chunk.await_args.args[0]["ts"] == ""


def test_search_surface_carries_the_speaker_but_never_doubles_it():
    """The retrieval surface names who spoke; CHUNK.text never changes.

    "I'm opening a dance studio" is a fact about Gina and names her nowhere, so
    stored verbatim it is unreachable by any question about Gina — the measured
    reason the prefixed arm led on knowledge_synthesis (0.736 vs 0.573) despite
    the clean arm holding MORE relations (72 vs 57). Prefixing the retrieval
    surface buys that reach without putting the name into the extractor's input,
    which is what mints speakers as ENTITY nodes.
    """
    from graphknows.ingestion.parsers.text import search_surface

    assert search_surface("I'm opening a dance studio.", "Gina") == (
        "Gina: I'm opening a dance studio."
    )
    # A caller that already flattened the speaker in must not get it twice.
    assert search_surface("Gina: I'm opening a dance studio.", "Gina") == (
        "Gina: I'm opening a dance studio."
    )
    assert search_surface("No speaker known.", "") == "No speaker known."
    assert search_surface("", "Gina") == ""


async def test_ingest_turn_embeds_the_surface_but_stores_text_verbatim():
    """The embedding is of "Gina: ..."; CHUNK.text stays exactly what was said."""
    embedded: list[str] = []
    handler, store, _ = _make_handler(
        extractor=None, embed_fn=lambda t: embedded.append(t) or [0.1] * 10
    )

    await handler.ingest_turn("s1", "f1", 0, "user", "I'm opening a dance studio.", speaker="Gina")

    assert embedded == ["Gina: I'm opening a dance studio."]
    chunk = store.write_chunk.await_args.args[0]
    assert chunk["text"] == "I'm opening a dance studio."
    assert chunk["search_text"] == "Gina: I'm opening a dance studio."


async def test_ingest_turn_logs_when_it_drops_an_empty_turn(caplog):
    """A dropped turn must leave a trace, or "not configured yet" and "broken" look alike.

    ``ingest_turn`` returns 0 for whitespace-only text. That is a defensible policy —
    what is not defensible is doing it silently: from outside, a caller that submitted
    200 turns and got 0 chunks cannot tell a content decision from a wedged extractor.
    CLAUDE.md prohibits the unlogged swallow by name ("If a soft-fail is right, log it").

    Asserts on the VALUE: the message must name the session and the sequence number, so
    the specific dropped turn is identifiable — a bare "skipped" counter would not have
    let anyone find which turn went missing.
    """
    import logging

    handler, store, _ = _make_handler(extractor=None)

    with caplog.at_level(logging.INFO, logger="graphknows.ingestion.stm.ingest"):
        written = await handler.ingest_turn("s-42", "f1", 7, "user", "   \n\t  ", speaker="Gina")

    assert written == 0
    store.write_chunk.assert_not_awaited()

    records = [r for r in caplog.records if "empty" in r.getMessage().lower()]
    assert records, (
        f"empty turn dropped with no log; saw {[r.getMessage() for r in caplog.records]}"
    )
    msg = records[0].getMessage()
    assert "s-42" in msg
    assert "7" in msg
