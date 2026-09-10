"""Per-transcript checkpointing by record ``uuid`` (FR-035)."""

from __future__ import annotations

import json
from pathlib import Path

from graphknows.integrations.claude_code.hooks import Checkpoints


def _write(transcript: Path, *records: dict[str, object]) -> None:
    transcript.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )


def _turn(uuid: str) -> dict[str, object]:
    return {"uuid": uuid, "type": "user", "message": {"content": uuid}}


def test_unread_offers_every_record_without_a_checkpoint(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write(transcript, _turn("a"), _turn("b"))

    unread = Checkpoints(tmp_path / "state.json").unread(transcript)

    assert [record["uuid"] for record in unread] == ["a", "b"]


def test_advance_makes_the_read_records_unread_no_longer(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write(transcript, _turn("a"), _turn("b"))
    checkpoints = Checkpoints(tmp_path / "state.json")

    checkpoints.advance(transcript, checkpoints.unread(transcript))
    _write(transcript, _turn("a"), _turn("b"), _turn("c"))

    assert [record["uuid"] for record in checkpoints.unread(transcript)] == ["c"]


def test_a_checkpoint_outlives_the_process(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    _write(transcript, _turn("a"))
    first = Checkpoints(state)
    first.advance(transcript, first.unread(transcript))

    assert Checkpoints(state).unread(transcript) == []


def test_a_checkpoint_is_kept_per_transcript(tmp_path: Path) -> None:
    one, other = tmp_path / "one.jsonl", tmp_path / "other.jsonl"
    _write(one, _turn("a"))
    _write(other, _turn("b"))
    checkpoints = Checkpoints(tmp_path / "state.json")

    checkpoints.advance(one, checkpoints.unread(one))

    assert [record["uuid"] for record in checkpoints.unread(other)] == ["b"]


def test_a_half_written_trailing_record_is_left_unread(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        f"{json.dumps(_turn('a'))}\n" + json.dumps(_turn("b"))[:12], encoding="utf-8"
    )
    checkpoints = Checkpoints(tmp_path / "state.json")

    unread = checkpoints.unread(transcript)
    checkpoints.advance(transcript, unread)
    _write(transcript, _turn("a"), _turn("b"))

    assert [record["uuid"] for record in unread] == ["a"]
    assert [record["uuid"] for record in checkpoints.unread(transcript)] == ["b"]


def test_a_trailing_record_without_a_uuid_is_left_unread(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write(transcript, _turn("a"), {"type": "user", "message": {"content": "b"}})
    checkpoints = Checkpoints(tmp_path / "state.json")

    checkpoints.advance(transcript, checkpoints.unread(transcript))

    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8")) == {
        str(transcript): "a"
    }


def test_an_unknown_checkpoint_offers_the_records_again(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({str(transcript): "gone"}), encoding="utf-8")
    _write(transcript, _turn("a"))

    assert [record["uuid"] for record in Checkpoints(state).unread(transcript)] == ["a"]


def test_advancing_on_nothing_keeps_the_earlier_checkpoint(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write(transcript, _turn("a"))
    checkpoints = Checkpoints(tmp_path / "state.json")
    checkpoints.advance(transcript, checkpoints.unread(transcript))

    checkpoints.advance(transcript, [])

    assert checkpoints.unread(transcript) == []
