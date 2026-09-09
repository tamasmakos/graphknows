"""Agent transcript JSONL: tolerant, versioned, and counted (FR-026, FR-027).

The harness's transcript format is internal and version-dependent, so this
parser recognises rather than validates: a record or block shape it does not
know is counted and skipped, never fatal and never silent. Records are read one
line at a time with stdlib ``json``, so a transcript still being appended to
costs one counted skip instead of an aborted ingest.
"""

from __future__ import annotations

import json
from collections import Counter as _Tally
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from graphknows.models.report import Counters
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source

KEPT_RECORD_TYPES = frozenset({"user", "assistant"})

# Block type to segment kind. ``tool_result`` is knowingly absent: it is a
# citable source, not evidence of its own (FR-028), so it becomes a
# :class:`~graphknows.models.source.Source` instead of a segment.
BLOCK_KINDS: Mapping[str, SegmentKind] = {
    "text": SegmentKind.turn,
    "thinking": SegmentKind.turn,
    "tool_use": SegmentKind.tool_call,
}
TOOL_RESULT_BLOCK = "tool_result"
TOOL_RESULT_MIME = "application/x-tool-result"


@dataclass(frozen=True)
class _Anchor:
    """Where one record sits: the source it belongs to and its byte span."""

    source: Source
    byte_range: tuple[int, int]


def _lines(data: bytes) -> Iterator[tuple[bytes, tuple[int, int]]]:
    """Each non-blank line of *data*, with its half-open byte span."""
    offset = 0
    for line in data.splitlines(keepends=True):
        if line.strip():
            yield line, (offset, offset + len(line))
        offset += len(line)


def _decode(line: bytes) -> dict[str, Any] | None:
    """The record on *line*, or ``None`` when it is not a JSON object."""
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _blocks(record: Mapping[str, Any]) -> list[Any] | None:
    """The record's content blocks, or ``None`` when the message is malformed.

    A plain-string content is the one-text-block shorthand the harness uses for
    a typed prompt.
    """
    message = record.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else None


def _timestamp(value: Any) -> datetime | None:
    """The record's ISO-8601 instant, or ``None`` when it has none to give."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _record_meta(record: Mapping[str, Any]) -> dict[str, str]:
    """The working directory, branch and side-chain flag carried by a record."""
    return {
        "cwd": str(record.get("cwd", "")),
        "gitBranch": str(record.get("gitBranch", "")),
        "isSidechain": str(bool(record.get("isSidechain", False))).lower(),
    }


def _tool_result_source(anchor: _Anchor, block: Mapping[str, Any], path: str) -> Source:
    """The tool result as a citable source of its own (FR-028).

    A result the harness left unidentified is still citable: the record path it
    sat at names it, so a fact never loses its citation to a naming quirk.
    """
    transcript = anchor.source
    tool_use_id = str(block.get("tool_use_id") or "")
    content = json.dumps(block.get("content", ""), sort_keys=True)
    return Source(
        uri=f"{transcript.uri}#{tool_use_id or path}",
        content_hash=sha256(content.encode()).hexdigest(),
        mime=TOOL_RESULT_MIME,
        namespace=transcript.namespace,
        meta={"transcript_source_id": transcript.id, "tool_use_id": tool_use_id},
    )


def _block_text(block: Mapping[str, Any]) -> str:
    """The citable text of one block: what was said, or what was called."""
    if "text" in block:
        return str(block["text"]).strip()
    if "thinking" in block:
        return str(block["thinking"]).strip()
    name = str(block.get("name", ""))
    return f"{name}({json.dumps(block.get('input', {}), sort_keys=True)})".strip()


class TranscriptParser:
    """Cuts an agent transcript into the turns and tool calls a fact can cite.

    ``counters`` reports the most recent parse only: a parser is reached through
    the registry once per source, and the counters belong to that source's
    ingest report rather than to the parser's lifetime.
    """

    mimes = frozenset({"application/x-ndjson", "application/jsonl"})
    version = "transcript/1"

    def __init__(self) -> None:
        self._tally: _Tally[str] = _Tally()
        self._tool_results: list[Source] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.version!r})"

    @property
    def counters(self) -> Counters:
        """The R9 counter set of the most recent parse."""
        return Counters(**self._tally)

    @property
    def tool_results(self) -> list[Source]:
        """The tool results of the most recent parse, as sources a fact can cite."""
        return list(self._tool_results)

    def parse(self, source: Source, data: bytes) -> list[Segment]:
        """The segments of *data*, in record order, counting everything skipped."""
        self._tally = _Tally()
        self._tool_results = []
        seen: set[str] = set()
        segments: list[Segment] = []
        for line, span in _lines(data):
            record = self._kept(line, seen)
            if record is None:
                continue
            segments.extend(self._segments(record, _Anchor(source, span)))
        return segments

    def _kept(self, line: bytes, seen: set[str]) -> dict[str, Any] | None:
        """The record on *line* if it is one to ingest, counting why it is not.

        ``isMeta`` marks the harness's own bookkeeping — expected, so dropped
        without a counter; drift is what the counters are for.
        """
        record = _decode(line)
        if record is None:
            self._tally["records_skipped_malformed"] += 1
            return None
        if record.get("type") not in KEPT_RECORD_TYPES:
            self._tally["records_skipped_unknown_type"] += 1
            return None
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            self._tally["records_skipped_malformed"] += 1
            return None
        if record.get("isMeta"):
            return None
        if uuid in seen:
            self._tally["records_skipped_duplicate"] += 1
            return None
        seen.add(uuid)
        return record

    def _segments(self, record: Mapping[str, Any], anchor: _Anchor) -> Iterator[Segment]:
        """Walk one record's content blocks into segments."""
        blocks = _blocks(record)
        if blocks is None:
            self._tally["records_skipped_malformed"] += 1
            return
        meta = _record_meta(record)
        for index, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                self._tally["blocks_skipped_unknown_type"] += 1
                continue
            block_type = block.get("type")
            path = f"{record['uuid']}/{index}"
            if block_type == TOOL_RESULT_BLOCK:
                self._tool_results.append(_tool_result_source(anchor, block, path))
                continue
            kind = BLOCK_KINDS.get(str(block_type))
            if kind is None:
                self._tally["blocks_skipped_unknown_type"] += 1
                continue
            text = _block_text(block)
            if not text:
                continue
            yield Segment(
                source_id=anchor.source.id,
                text=text,
                kind=kind,
                path=path,
                byte_range=anchor.byte_range,
                role=str(record.get("type", "")),
                observed_at=_timestamp(record.get("timestamp")),
                meta=meta,
            )


__all__ = ["TranscriptParser"]
