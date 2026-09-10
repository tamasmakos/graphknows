"""Claude Code hooks: stdlib-only, and never reading a transcript twice (FR-035).

A transcript is still being appended to while the session that writes it fires
its stop and pre-compaction hooks, so its trailing line can be half-written.
Reading stops at the first line that is not a whole record carrying a ``uuid``,
and the checkpoint only ever names a record that was read whole — the half-line
is picked up on the next event, once the harness has finished writing it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def _decode(line: bytes) -> Record | None:
    """The record on *line*, or ``None`` when it is not whole and identified."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("uuid"), str):
        return None
    return record


def _whole_records(data: bytes) -> list[Record]:
    """The records of *data* up to the first one that is not whole.

    A transcript is append-only, so the first malformed line is the line being
    written: stopping there costs one event's delay, never a lost record.
    """
    records: list[Record] = []
    for line in data.splitlines():
        if not line.strip():
            continue
        record = _decode(line)
        if record is None:
            break
        records.append(record)
    return records


def _load(path: Path) -> dict[str, str]:
    """The checkpoints stored at *path*, or none when it has none to give."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {str(key): str(value) for key, value in stored.items()}


class Checkpoints:
    """Where each transcript was last read to, named by record ``uuid``.

    A uuid the transcript no longer carries — a rotated or replaced file — is
    treated as no checkpoint at all, so the records are offered again rather
    than silently dropped.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen = _load(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._path)!r})"

    def unread(self, transcript: Path) -> list[Record]:
        """The whole records of *transcript* written after its checkpoint."""
        records = _whole_records(transcript.read_bytes())
        seen = self._seen.get(str(transcript))
        for index, record in enumerate(records):
            if record["uuid"] == seen:
                return records[index + 1 :]
        return records

    def advance(self, transcript: Path, records: Sequence[Record]) -> None:
        """Mark *records* of *transcript* as read, keeping the earlier mark if empty."""
        if not records:
            return
        self._seen[str(transcript)] = str(records[-1]["uuid"])
        self._path.write_text(json.dumps(self._seen, sort_keys=True), encoding="utf-8")
