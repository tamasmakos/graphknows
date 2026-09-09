"""Tool results are citable sources, not segments (FR-028)."""

from __future__ import annotations

import json
from typing import Any

from graphknows.ingestion.parsers.transcript import TOOL_RESULT_MIME, TranscriptParser
from graphknows.models.source import Source
from graphknows.retrieval.retriever import _facts_with_evidence

SOURCE = Source(
    uri="session.jsonl", content_hash="abc", mime="application/x-ndjson", namespace="agent"
)


def _jsonl(*blocks: dict[str, Any]) -> bytes:
    """One assistant record carrying *blocks*, as transcript bytes."""
    record = {
        "type": "assistant",
        "uuid": "u1",
        "timestamp": "2026-09-08T10:11:12.000Z",
        "message": {"role": "assistant", "content": list(blocks)},
    }
    return json.dumps(record).encode()


def test_a_tool_result_becomes_a_source_a_fact_can_cite_not_a_segment():
    parser = TranscriptParser()
    data = _jsonl(
        {"type": "text", "text": "Read it."},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
    )

    segments = parser.parse(SOURCE, data)

    assert [s.text for s in segments] == ["Read it."]
    (result,) = parser.tool_results
    assert result.uri == "session.jsonl#toolu_1"
    assert result.mime == TOOL_RESULT_MIME
    assert result.namespace == SOURCE.namespace
    assert result.meta["transcript_source_id"] == SOURCE.id
    assert parser.counters.blocks_skipped_unknown_type == 0


def test_a_tool_result_the_harness_left_unidentified_is_still_citable():
    parser = TranscriptParser()

    parser.parse(SOURCE, _jsonl({"type": "tool_result", "content": "ok"}))

    assert [r.uri for r in parser.tool_results] == ["session.jsonl#u1/0"]


def test_the_same_result_content_at_the_same_call_is_the_same_source():
    parser = TranscriptParser()
    block = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}

    parser.parse(SOURCE, _jsonl(block))
    first = parser.tool_results[0]
    parser.parse(SOURCE, _jsonl(block))

    assert parser.tool_results[0].id == first.id


def test_tool_results_report_the_most_recent_parse_only():
    parser = TranscriptParser()
    parser.parse(SOURCE, _jsonl({"type": "tool_result", "content": "ok"}))

    parser.parse(SOURCE, _jsonl({"type": "text", "text": "no tools here"}))

    assert parser.tool_results == []


def test_recall_survives_a_citation_to_a_tool_result_that_was_never_ingested():
    """The tool-result source is citable even when its segment is not there."""
    row = {
        "id": "f1",
        "subject": "e1",
        "predicate": "touched",
        "object": "a.py",
        "confidence": 0.9,
        "source_uri": "session.jsonl#toolu_1",
    }

    (recalled,) = _facts_with_evidence([row], max_evidence=3)

    assert recalled.fact.id == "f1"
    assert recalled.evidence[0].source_uri == "session.jsonl#toolu_1"
    assert recalled.evidence[0].text == ""
    assert recalled.evidence[0].byte_range == (0, 0)
