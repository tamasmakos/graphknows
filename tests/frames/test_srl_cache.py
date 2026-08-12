"""The sentence cache: overlapping windows must not re-parse the same turn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphknows.frames import srl


class _Parser:
    """Counts what it was asked to parse; one Motion frame per 'went'."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def detect_frames_bulk(self, sents: list[str]) -> list[SimpleNamespace]:
        self.seen.append(list(sents))
        out = []
        for s in sents:
            frames = []
            if (i := s.find("went")) >= 0:
                frames.append(
                    SimpleNamespace(
                        name="Motion",
                        trigger_location=i,
                        frame_elements=[SimpleNamespace(name="Theme", text="I")],
                    )
                )
            out.append(SimpleNamespace(frames=frames))
        return out


@pytest.fixture
def parser(monkeypatch: pytest.MonkeyPatch) -> _Parser:
    p = _Parser()
    srl._sentence_cache.clear()
    monkeypatch.setattr(srl, "_load", lambda: p)
    return p


def test_a_repeated_turn_is_parsed_once(parser: _Parser) -> None:
    # Two overlapping windows sharing a turn — the stride-3 case.
    srl.parse("Caroline: I went home.\nMel: Nice.")
    srl.parse("Mel: Nice.\nCaroline: I went home.\nMel: Again?")

    parsed = [s for batch in parser.seen for s in batch]
    assert parsed.count("Caroline: I went home.") == 1
    assert parsed.count("Mel: Nice.") == 1
    assert "Mel: Again?" in parsed


def test_a_duplicate_within_one_chunk_is_parsed_once(parser: _Parser) -> None:
    srl.parse("Mel: Nice.\nMel: Nice.")
    assert parser.seen == [["Mel: Nice."]]


def test_cached_hits_are_rebased_onto_the_asking_chunk(parser: _Parser) -> None:
    first = srl.parse("Caroline: I went home.")
    text = "Mel: Hi.\nCaroline: I went home."
    second = srl.parse(text)

    assert first[0].offset == "Caroline: I went home.".index("went")
    assert second[0].offset == text.index("went")
    # ...and the FE span travels with it.
    role, start, end = second[0].fes[0]
    assert text[start:end] == "I"


def test_a_full_cache_is_dropped_rather_than_grown(
    parser: _Parser, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(srl, "_CACHE_MAX", 1)
    srl.parse("Mel: one.")
    srl.parse("Mel: two.")
    assert len(srl._sentence_cache) <= 1
