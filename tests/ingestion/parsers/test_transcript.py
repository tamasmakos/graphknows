"""The transcript parser: what it keeps, what it retains, and what it counts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from graphknows.ingestion.parsers.registry import Parser
from graphknows.ingestion.parsers.transcript import TranscriptParser
from graphknows.models.segment import SegmentKind
from graphknows.models.source import Source

SOURCE = Source(uri="session.jsonl", content_hash="abc", mime="application/x-ndjson")


def _record(uuid: str, blocks: list[dict[str, Any]] | str, **extra: Any) -> dict[str, Any]:
    """One transcript record in the harness's shape, with sane defaults."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": "2026-09-08T10:11:12.000Z",
        "cwd": "/repo",
        "gitBranch": "main",
        "isSidechain": False,
        "message": {"role": "assistant", "content": blocks},
        **extra,
    }


def _jsonl(*records: Any) -> bytes:
    return "\n".join(json.dumps(r) for r in records).encode()


def test_the_parser_is_the_registry_seam():
    assert isinstance(TranscriptParser(), Parser)


def test_text_thinking_and_tool_use_become_segments_but_a_tool_result_does_not():
    parser = TranscriptParser()
    data = _jsonl(
        _record(
            "u1",
            [
                {"type": "text", "text": "Shipping the parser."},
                {"type": "thinking", "thinking": "It must not be fatal."},
                {"type": "tool_use", "name": "Read", "input": {"path": "a.py"}},
                {"type": "tool_result", "content": "ok"},
            ],
        )
    )

    segments = parser.parse(SOURCE, data)

    assert [s.kind for s in segments] == [
        SegmentKind.turn,
        SegmentKind.turn,
        SegmentKind.tool_call,
    ]
    assert segments[0].text == "Shipping the parser."
    assert segments[2].text == 'Read({"path": "a.py"})'
    assert parser.counters.blocks_skipped_unknown_type == 0


def test_a_record_retains_its_time_directory_branch_and_side_chain_flag():
    data = _jsonl(_record("u1", "hello", type="user", isSidechain=True))

    segment = TranscriptParser().parse(SOURCE, data)[0]

    assert segment.observed_at == datetime(2026, 9, 8, 10, 11, 12, tzinfo=UTC)
    assert segment.observed_at_inferred is False
    assert segment.meta == {"cwd": "/repo", "gitBranch": "main", "isSidechain": "true"}
    assert segment.role == "user"
    assert segment.source_id == SOURCE.id


def test_every_segment_addresses_the_bytes_of_its_own_record():
    data = _jsonl(_record("u1", "first"), _record("u2", "second"))

    first, second = TranscriptParser().parse(SOURCE, data)

    assert data[slice(*first.byte_range)].endswith(b"}\n")
    assert json.loads(data[slice(*second.byte_range)])["uuid"] == "u2"
    assert first.id != second.id


def test_an_unknown_record_type_is_counted_and_skipped():
    parser = TranscriptParser()
    data = _jsonl(
        {"type": "file-history-snapshot", "uuid": "s1"},
        {"type": "mode", "mode": "normal"},
        _record("u1", "kept"),
    )

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["kept"]
    assert parser.counters.records_skipped_unknown_type == 2


def test_an_unknown_block_type_is_counted_and_skipped():
    parser = TranscriptParser()
    data = _jsonl(
        _record("u1", [{"type": "hologram", "pixels": []}, {"type": "text", "text": "kept"}])
    )

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["kept"]
    assert parser.counters.blocks_skipped_unknown_type == 1


def test_a_malformed_record_is_counted_and_never_fatal():
    parser = TranscriptParser()
    data = _jsonl(_record("u1", "kept")) + b'\n{"type": "user", "uuid": "u2", "mess'

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["kept"]
    assert parser.counters.records_skipped_malformed == 1


def test_a_record_without_a_message_is_malformed_not_a_crash():
    parser = TranscriptParser()

    assert parser.parse(SOURCE, _jsonl({"type": "user", "uuid": "u1"})) == []
    assert parser.counters.records_skipped_malformed == 1


def test_a_repeated_record_identifier_is_counted_and_parsed_once():
    parser = TranscriptParser()
    data = _jsonl(_record("u1", "once"), _record("u1", "once"))

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["once"]
    assert parser.counters.records_skipped_duplicate == 1


def test_internal_bookkeeping_is_dropped_without_counting_as_drift():
    parser = TranscriptParser()
    data = _jsonl(_record("u1", "caveat", type="user", isMeta=True), _record("u2", "kept"))

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["kept"]
    assert parser.counters.records_skipped_unknown_type == 0
    assert parser.counters.records_skipped_malformed == 0


def test_counters_report_the_most_recent_parse_only():
    parser = TranscriptParser()
    parser.parse(SOURCE, _jsonl({"type": "mode"}))
    assert parser.counters.records_skipped_unknown_type == 1

    parser.parse(SOURCE, _jsonl(_record("u1", "kept")))
    assert parser.counters.records_skipped_unknown_type == 0
