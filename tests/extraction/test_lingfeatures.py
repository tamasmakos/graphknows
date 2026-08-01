"""Guard tests for the linguistic helpers left behind by the SVO removal.

``relations/svo.py`` mined relations; that miner is gone (measured on a live
conv-30 graph it produced 28 of the 43 relation edges and 25 of those 28 were
``<speaker> HAS <common noun>`` — ``Jon HAS back``, ``Jon HAS corner``). Three
helpers in that module were never relation mining and still have callers, so
they moved to ``relations/_lingfeatures.py``:

* ``_resolve``       — speaker resolution for frame-SRL role fillers
* ``noun_supersense``— the WordNet type fallback for relation arguments
* ``parse_speaker``  — the LoCoMo ``"ts | Speaker: text"`` turn-line parser
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphknows.ingestion.extraction.relations._lingfeatures import (
    _resolve,
    noun_supersense,
    parse_speaker,
    resolve_deixis,
)


def _tok(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def test_first_person_resolves_to_the_speaker():
    assert _resolve(_tok("I"), "Caroline", ["Caroline", "Melanie"]) == "Caroline"
    assert _resolve(_tok("my"), "Caroline", ["Caroline", "Melanie"]) == "Caroline"


def test_second_person_resolves_to_the_other_of_two_speakers():
    assert _resolve(_tok("you"), "Caroline", ["Caroline", "Melanie"]) == "Melanie"


def test_second_person_is_left_alone_unless_there_are_exactly_two_speakers():
    assert _resolve(_tok("you"), "Caroline", ["Caroline"]) == "you"
    assert _resolve(_tok("you"), "", ["Caroline", "Melanie"]) == "you"


def test_third_person_surface_is_returned_unchanged():
    assert _resolve(_tok("Melanie"), "Caroline", ["Caroline", "Melanie"]) == "Melanie"


def test_noun_supersense_reads_the_lemmatised_head_noun():
    assert noun_supersense("dance studio") == "noun.artifact"
    assert noun_supersense("nurse") == "noun.person"
    assert noun_supersense("motorcycles") == "noun.artifact"  # lemmatised to motorcycle
    assert noun_supersense("") == ""


def test_parse_speaker():
    assert parse_speaker("8:30 pm on 12 May, 2023 | Jon: I lost my job.") == "Jon"
    assert parse_speaker("Caroline: hi there") == "Caroline"
    assert parse_speaker("a line with no speaker prefix") == ""


# --- the deixis-resolved relation surface ------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Tokenizer-only spaCy pipeline: resolve_deixis reads text/whitespace_/idx."""
    import spacy

    return spacy.blank("en")


def test_first_person_subject_becomes_the_speaker(tokenizer):
    """The measured lever: relex needs two entity SPANS and "I" is not one.

    40 real conv-30 turns, same model/labels/threshold: 3 relations from the raw
    text, 19 from the resolved surface.
    """
    out = resolve_deixis(
        tokenizer("1:56 pm on 8 May, 2023 | Gina: I opened an online clothing store."), ""
    )

    assert "Gina opened an online clothing store" in out


def test_a_contraction_is_expanded_not_mangled(tokenizer):
    """A regex over raw text turns "I'm over the moon" into "gina is over the moon be".

    Rebuilding token-wise leaves "Gina'm", which is not a word either — the
    clitic following a replaced pronoun has to become its full 3rd-person form.
    """
    out = resolve_deixis(tokenizer("Gina: I'm over the moon and I've been teaching dance."), "")

    assert "Gina is over the moon" in out
    assert "Gina has been teaching dance" in out
    assert "'m" not in out and "'ve" not in out


def test_a_possessive_keeps_its_genitive_marker(tokenizer):
    out = resolve_deixis(tokenizer("Gina: My collection has twenty pieces."), "")

    assert "Gina's collection has twenty pieces" in out


def test_second_person_resolves_to_the_other_speaker_of_the_chunk(tokenizer):
    text = "Gina: Do you still teach?\nJon: I do, and your studio inspired me."
    out = resolve_deixis(tokenizer(text), "")

    assert "Do Jon still teach" in out
    assert "Gina's studio inspired Jon" in out


def test_the_speaker_argument_covers_lines_with_no_turn_prefix(tokenizer):
    out = resolve_deixis(tokenizer("I opened an online clothing store."), "Gina")

    assert out.startswith("Gina opened")


def test_no_known_speaker_returns_empty_so_the_caller_falls_back(tokenizer):
    assert resolve_deixis(tokenizer("I opened an online clothing store."), "") == ""


def test_text_without_deixis_round_trips_verbatim(tokenizer):
    text = "Gina: Melanie opened an online clothing store in Denver, right?"
    assert resolve_deixis(tokenizer(text), "") == text
