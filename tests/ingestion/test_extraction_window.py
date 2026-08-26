"""The extraction window: what the extractor reads vs what the chunk keeps.

The window widens the extractor's view across turn boundaries without widening
what memory stores. Two things must hold for that to be true rather than merely
plausible, and both are silent when broken:

- the stored chunk is still the single turn (a window that leaks into CHUNK.text
  double-stores every utterance and corrupts retrieval), and
- entities from the earlier turns do not become THIS chunk's entities (they were
  already minted against their own chunks; re-minting them hangs MENTIONS edges
  on the wrong chunk and re-floods the entity channel).
"""

from __future__ import annotations

from graphknows.ingestion.stm.ingest import _entities_from, _STMIngestHandler


def _handler(window: int) -> _STMIngestHandler:
    handler = _STMIngestHandler.__new__(_STMIngestHandler)  # no store/extractor needed
    handler._extract_window = window
    handler._recent_turns = {}
    return handler


def _window(handler: _STMIngestHandler, session: str, text: str, speaker: str) -> tuple[str, int]:
    """``(text, offset)`` — the per-turn spans are exercised in test_lexical_labels."""
    rendered, offset, _spans, _verbatim = handler._extraction_window(session, text, speaker)
    return rendered, offset


def test_first_turn_is_its_own_window() -> None:
    """Nothing precedes it, so the text is the turn verbatim and the offset is 0."""
    text, offset = _window(_handler(5), "s1", "I adopted a dog.", "Gina")
    assert (text, offset) == ("I adopted a dog.", 0)


def test_window_of_one_is_an_exact_replay() -> None:
    """The control arm must be byte-identical, not merely similar."""
    handler = _handler(1)
    _window(handler, "s1", "I adopted a dog.", "Gina")
    text, offset = _window(handler, "s1", "What did you name her?", "Jon")
    assert (text, offset) == ("What did you name her?", 0)


def test_later_turn_carries_the_attributed_prefix() -> None:
    handler = _handler(5)
    _window(handler, "s1", "I adopted a dog.", "Gina")
    text, offset = _window(handler, "s1", "What did you name her?", "Jon")

    # Every line is speaker-attributed now, including the current turn — deixis
    # rewrites that line too, so it needs the same attribution the prefix has.
    assert text == "Gina: Gina adopted a dog.\nJon: What did Gina name her?"
    # The offset must land exactly on the current turn, or the entity filter
    # silently keeps or drops the wrong spans.
    assert text[offset:] == "What did Gina name her?"


def test_window_holds_at_most_the_configured_turns() -> None:
    handler = _handler(3)
    for i in range(5):
        text, _ = _window(handler, "s1", f"turn {i}", "Gina")
    # window=3 means this turn plus the 2 before it.
    assert text == "Gina: turn 2\nGina: turn 3\nGina: turn 4"


def test_sessions_do_not_bleed_into_each_other() -> None:
    handler = _handler(5)
    _window(handler, "s1", "I adopted a dog.", "Gina")
    text, offset = _window(handler, "s2", "Different conversation.", "Jon")
    assert (text, offset) == ("Different conversation.", 0)


def test_unextracted_turns_still_provide_context() -> None:
    """A greeting mints nothing but still establishes who is being addressed."""
    handler = _handler(5)
    _window(handler, "s1", "Hey Jon!", "Gina")
    text, _ = _window(handler, "s1", "How is the studio?", "Jon")
    assert text.startswith("Gina: Hey Jon!")


def test_prior_turn_entities_are_dropped_and_spans_rebased() -> None:
    offset = len("Gina: I adopted a dog.\n")
    entities = [
        {"name": "Gina", "start": 0, "end": 4},  # prefix: already minted elsewhere
        {"name": "Mochi", "start": offset + 18, "end": offset + 23},
    ]

    kept = _entities_from(entities, offset)

    assert [e["name"] for e in kept] == ["Mochi"]
    # Rebased onto the stored chunk, not the window it was found in.
    assert kept[0]["start"] == 18


def test_an_entity_without_offsets_is_kept() -> None:
    """Unattributable to an earlier turn any more than to this one; dropping is worse."""
    kept = _entities_from([{"name": "Mochi", "start": -1, "end": -1}], 20)
    assert [e["name"] for e in kept] == ["Mochi"]
