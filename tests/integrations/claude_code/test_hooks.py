"""The six hook verbs against a fake memory service.

The service is a recorder, not the real ``graphknows-mcp``: what is under test
is the verbs' payload → tool-call → output-block mapping, above all the rule
that nothing is injected when no symbol resolves (FR-031, SC-009).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graphknows.integrations.claude_code import hooks


class FakeService:
    """Records tool calls and answers ``memory_recall`` with a canned result."""

    def __init__(self, recall_result: Any) -> None:
        self.recall_result = recall_result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Record the call and answer it the way the service would."""
        self.calls.append((name, arguments or {}))
        return self.recall_result if name == "memory_recall" else {"file_id": "s1"}


def one_fact(text: str) -> dict[str, Any]:
    """A ``memory_recall`` result whose single fact is evidenced by *text*."""
    evidence = {"source_uri": "file:///repo/a.py", "byte_range": [0, len(text)], "text": text}
    return {"facts": [{"fact": {"id": "f1"}, "evidence": [evidence]}], "no_evidence": False}


# "Nothing is known", as recall states it (FR-015) — not an empty list left to
# be read as a successful answer.
NOTHING_KNOWN = {"facts": [], "no_evidence": True}

# One payload carrying every field the recall verbs read, so each verb has
# something to resolve and "nothing injected" is the service's answer, not a
# missing key.
FULL_EVENT = {
    "cwd": "/repo",
    "prompt": "nothing here resolves",
    "tool_input": {"file_path": "a.py"},
    "tool_response": "class Unknown",
}


def write_transcript(path: Path, *records: dict[str, Any]) -> None:
    """Write *records* as a JSONL transcript at *path*."""
    path.write_text("".join(f"{json.dumps(r)}\n" for r in records), encoding="utf-8")


def turn(uuid: str, text: str) -> dict[str, Any]:
    """One user record of a transcript, saying *text*."""
    return {
        "uuid": uuid,
        "type": "user",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": text},
    }


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A two-record transcript, with the checkpoint state kept beside it."""
    monkeypatch.setenv("GRAPHKNOWS_HOOK_STATE", str(tmp_path / "state.json"))
    path = tmp_path / "s1.jsonl"
    write_transcript(path, turn("a", "first thing said"), turn("b", "second thing said"))
    return path


@pytest.mark.parametrize(
    ("verb", "event", "expected_event_name", "expected_query"),
    [
        ("context", {"cwd": "/repo"}, "SessionStart", "/repo"),
        ("recall", {"prompt": "what about Acme"}, "UserPromptSubmit", "what about Acme"),
        ("preview", {"tool_input": {"file_path": "a.py"}}, "PreToolUse", "a.py"),
        ("observe", {"tool_response": "class Segment"}, "PostToolUse", "class Segment"),
    ],
)
async def test_recall_verbs_inject_resolved_facts(
    verb: str, event: dict[str, Any], expected_event_name: str, expected_query: str
) -> None:
    service = FakeService(one_fact("Acme is Platinum"))

    output = await hooks.run_verb(verb, event, service)

    assert service.calls == [("memory_recall", {"query": expected_query})]
    assert output == {
        "hookSpecificOutput": {
            "hookEventName": expected_event_name,
            "additionalContext": "- Acme is Platinum",
        }
    }


@pytest.mark.parametrize("verb", ["context", "recall", "preview", "observe"])
async def test_nothing_is_injected_when_no_symbol_resolves(verb: str) -> None:
    service = FakeService(NOTHING_KNOWN)

    assert await hooks.run_verb(verb, FULL_EVENT, service) is None
    assert [name for name, _ in service.calls] == ["memory_recall"]


@pytest.mark.parametrize("verb", ["context", "recall", "preview", "observe"])
async def test_an_empty_payload_never_reaches_the_service(verb: str) -> None:
    service = FakeService(one_fact("never asked for"))

    assert await hooks.run_verb(verb, {}, service) is None
    assert service.calls == []


@pytest.mark.parametrize("verb", ["remember", "catchup"])
async def test_ingest_verbs_ingest_the_transcript_and_inject_nothing(
    verb: str, transcript: Path
) -> None:
    service = FakeService(NOTHING_KNOWN)
    event = {"transcript_path": str(transcript), "session_id": "s1", "cwd": "/repo"}

    assert await hooks.run_verb(verb, event, service) is None
    assert service.calls == [
        (
            "memory_ingest",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "first thing said",
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    {
                        "role": "user",
                        "content": "second thing said",
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                ],
                "session_id": "s1",
            },
        )
    ]


async def test_remember_ingests_only_what_is_new_since_the_checkpoint(transcript: Path) -> None:
    event = {"transcript_path": str(transcript), "session_id": "s1"}
    await hooks.run_verb("remember", event, FakeService(NOTHING_KNOWN))
    write_transcript(
        transcript,
        turn("a", "first thing said"),
        turn("b", "second thing said"),
        turn("c", "third thing said"),
    )
    service = FakeService(NOTHING_KNOWN)

    await hooks.run_verb("remember", event, service)

    (_, arguments), *rest = service.calls
    assert not rest
    assert [message["content"] for message in arguments["messages"]] == ["third thing said"]


async def test_catchup_ingests_the_whole_transcript_past_the_checkpoint(transcript: Path) -> None:
    event = {"transcript_path": str(transcript), "session_id": "s1"}
    await hooks.run_verb("remember", event, FakeService(NOTHING_KNOWN))
    service = FakeService(NOTHING_KNOWN)

    await hooks.run_verb("catchup", event, service)

    (_, arguments), *rest = service.calls
    assert not rest
    assert [message["content"] for message in arguments["messages"]] == [
        "first thing said",
        "second thing said",
    ]


async def test_tool_records_are_cited_not_remembered(transcript: Path) -> None:
    write_transcript(
        transcript,
        {"uuid": "a", "type": "system", "message": {"role": "system", "content": "boot"}},
        {
            "uuid": "b",
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "reading it"},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                    {"type": "tool_result", "content": "file contents"},
                ],
            },
        },
    )
    service = FakeService(NOTHING_KNOWN)

    await hooks.run_verb("remember", {"transcript_path": str(transcript)}, service)

    (_, arguments), *rest = service.calls
    assert not rest
    assert [message["content"] for message in arguments["messages"]] == ["reading it"]


@pytest.mark.parametrize("verb", ["remember", "catchup"])
async def test_ingest_verbs_without_a_transcript_do_nothing(verb: str) -> None:
    service = FakeService(NOTHING_KNOWN)

    assert await hooks.run_verb(verb, {"session_id": "s1"}, service) is None
    assert service.calls == []


def test_main_rejects_an_unknown_verb(capsys: pytest.CaptureFixture[str]) -> None:
    assert hooks.main(["forget-everything"]) == 2
    assert "usage: hooks" in capsys.readouterr().err
