"""Claude Code hooks: stdlib-only, and never reading a transcript twice (FR-035).

``python -m graphknows.integrations.claude_code.hooks <verb>`` reads a hook
event as JSON on stdin and writes a ``hookSpecificOutput`` block on stdout, or
nothing at all. Injection is symbol-keyed (FR-031): only facts the query's
symbols activate are injected, and a query resolving to no symbol injects
nothing (SC-009). Memory is reached only through
:mod:`graphknows.integrations.client`, so the hook process loads no
machine-learning dependency (FR-036).

A transcript is still being appended to while the session that writes it fires
its stop and pre-compaction hooks, so its trailing line can be half-written.
Reading stops at the first line that is not whole JSON, and the checkpoint only
ever names a record that was read whole — the half-line is picked up on the
next event, once the harness has finished writing it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graphknows.integrations.client import GraphKnowsMCPClient

Record = dict[str, Any]


def _decode(line: bytes) -> Record | None:
    """The record on *line*, or ``None`` when it is not a whole JSON object."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _whole_records(data: bytes) -> list[Record]:
    """The identified records of *data* up to the first one that is not whole.

    A transcript is append-only, so the first unparseable line is the line
    being written: stopping there costs one event's delay, never a lost record.
    A record the harness writes without a ``uuid`` — ``mode``, ``last-prompt``,
    ``file-history-snapshot`` and the other session-state lines — is whole but
    unidentified, so it is skipped rather than read as the end of the file.
    """
    records: list[Record] = []
    for line in data.splitlines():
        if not line.strip():
            continue
        record = _decode(line)
        if record is None:
            break
        if isinstance(record.get("uuid"), str):
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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._seen, sort_keys=True), encoding="utf-8")


class MemoryService(Protocol):
    """The slice of the client SDK a verb uses."""

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call the MCP tool *name* with *arguments* and return its result."""
        ...


def _state_path() -> Path:
    """Where the per-transcript checkpoints live."""
    default = Path.home() / ".graphknows" / "hook-checkpoints.json"
    return Path(os.environ.get("GRAPHKNOWS_HOOK_STATE") or default)


async def _recall_about(service: MemoryService, query: str) -> str | None:
    """Recall the facts *query*'s symbols activate, or ``None`` for nothing known."""
    if not query.strip():
        return None
    result = await service.call_tool("memory_recall", {"query": query})
    return _render(result)


def _evidence_texts(fact: Any) -> list[str]:
    """The quoted segment texts of one ``FactWithEvidence`` payload."""
    if not isinstance(fact, dict):
        return []
    evidence = fact.get("evidence") or []
    return [
        text.strip()
        for item in evidence
        if isinstance(item, dict) and isinstance(text := item.get("text"), str) and text.strip()
    ]


def _render(result: Any) -> str | None:
    """Render a ``RecallResult`` payload as context lines, ``None`` when nothing is known.

    Only facts the query's symbols activated are injected, quoted at the
    evidence that asserts them (FR-009): a recall that says ``no_evidence``
    injects nothing, so injection stays symbol-keyed (FR-031).
    """
    if not isinstance(result, dict):
        return None
    lines = [f"- {text}" for fact in result.get("facts") or [] for text in _evidence_texts(fact)]
    return "\n".join(lines) or None


def _response_text(response: Any) -> str:
    """Flatten a ``tool_response`` payload to the text the symbols were seen in."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return " ".join(value for value in response.values() if isinstance(value, str))
    return ""


# What a record says, as opposed to what it did: tool calls and their results
# are citable sources, not evidence of their own (FR-028).
_REMEMBERED_TYPES = frozenset({"user", "assistant"})
_SPOKEN_BLOCKS = frozenset({"text", "thinking"})


def _spoken_text(content: Any) -> str:
    """What was said in a message's ``content``, tool blocks left out."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    spoken = (
        block.get("text") or block.get("thinking")
        for block in content
        if isinstance(block, dict) and block.get("type") in _SPOKEN_BLOCKS
    )
    return "\n".join(text.strip() for text in spoken if isinstance(text, str) and text.strip())


def _message(record: Record) -> dict[str, str] | None:
    """*record* as a chat message, or ``None`` when it says nothing to remember."""
    if record.get("type") not in _REMEMBERED_TYPES:
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    content = _spoken_text(message.get("content"))
    if not content:
        return None
    return {
        "role": str(message.get("role") or record["type"]),
        "content": content,
        "timestamp": str(record.get("timestamp") or ""),
    }


async def _ingest_transcript(service: MemoryService, event: dict[str, Any], *, full: bool) -> None:
    """Ingest the event's transcript; ``full`` ignores the checkpoint and catches up.

    The checkpoint advances only once the service has taken the records, so a
    failed ingest is offered again rather than lost.
    """
    transcript_path = event.get("transcript_path")
    if not transcript_path:
        return
    transcript = Path(str(transcript_path))
    if not transcript.is_file():
        return
    checkpoints = Checkpoints(_state_path())
    records = _whole_records(transcript.read_bytes()) if full else checkpoints.unread(transcript)
    messages = [message for record in records if (message := _message(record))]
    if messages:
        await service.call_tool(
            "memory_ingest",
            {"messages": messages, "session_id": str(event.get("session_id") or "")},
        )
    checkpoints.advance(transcript, records)


async def context(event: dict[str, Any], service: MemoryService) -> str | None:
    """SessionStart: what is known about the working directory."""
    return await _recall_about(service, str(event.get("cwd", "")))


async def recall(event: dict[str, Any], service: MemoryService) -> str | None:
    """UserPromptSubmit: facts attached to the symbols the prompt resolves to."""
    return await _recall_about(service, str(event.get("prompt", "")))


async def preview(event: dict[str, Any], service: MemoryService) -> str | None:
    """PreToolUse: what is known about the path an edit is about to touch."""
    tool_input = event.get("tool_input")
    path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
    return await _recall_about(service, str(path))


async def observe(event: dict[str, Any], service: MemoryService) -> str | None:
    """PostToolUse: what is known about the symbols a read just surfaced."""
    return await _recall_about(service, _response_text(event.get("tool_response")))


async def remember(event: dict[str, Any], service: MemoryService) -> None:
    """Stop: ingest the transcript records new since the last checkpoint."""
    await _ingest_transcript(service, event, full=False)
    return None


async def catchup(event: dict[str, Any], service: MemoryService) -> None:
    """PreCompact: ingest the whole transcript before the window is rewritten."""
    await _ingest_transcript(service, event, full=True)
    return None


@dataclass(frozen=True)
class Verb:
    """One hook verb: the event it answers and the coroutine that answers it."""

    event_name: str
    run: Callable[[dict[str, Any], MemoryService], Awaitable[str | None]]


VERBS: dict[str, Verb] = {
    "context": Verb("SessionStart", context),
    "recall": Verb("UserPromptSubmit", recall),
    "preview": Verb("PreToolUse", preview),
    "observe": Verb("PostToolUse", observe),
    "remember": Verb("Stop", remember),
    "catchup": Verb("PreCompact", catchup),
}


async def run_verb(
    name: str, event: dict[str, Any], service: MemoryService
) -> dict[str, Any] | None:
    """Run one verb and return its hook output block, or None when nothing is injected."""
    verb = VERBS[name]
    additional_context = await verb.run(event, service)
    if not additional_context:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": verb.event_name,
            "additionalContext": additional_context,
        }
    }


async def _run_against_service(name: str, event: dict[str, Any]) -> dict[str, Any] | None:
    """Run one verb against the running memory service."""
    command = os.environ.get("GRAPHKNOWS_MCP_COMMAND", "graphknows-mcp")
    async with GraphKnowsMCPClient(command=command) as service:
        return await run_verb(name, event, service)


def main(argv: list[str] | None = None) -> int:
    """Read the hook event from stdin, run the named verb, print its output block."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in VERBS:
        print(f"usage: hooks {'|'.join(VERBS)}", file=sys.stderr)
        return 2
    event = json.loads(sys.stdin.read() or "{}")
    output = asyncio.run(_run_against_service(args[0], event))
    if output is not None:
        print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
