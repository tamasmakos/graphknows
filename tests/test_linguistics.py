"""Modality/polarity/epistemic rules, sentence by sentence (plan §11.8).

This is the one stage whose errors are invisible in aggregate metrics: a wrong
``epistemic`` does not fail an ingest, does not move evidence_recall, and only
shows up as a confidently wrong answer much later. So every rule gets a sentence
with its expected value written out.

The trigger offset is located by searching for the trigger word, which is how
the SRL path supplies it.
"""

from __future__ import annotations

import pytest

from graphknows.linguistics import analyse

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def nlp():
    import spacy

    return spacy.load("en_core_web_lg", disable=["ner"])


def _run(nlp, text: str, trigger: str):
    doc = nlp(text)
    return analyse(doc, text.index(trigger))


# (text, trigger, field, expected)
CASES = [
    # --- polarity ---
    ("Melanie moved to Berlin.", "moved", "polarity", "pos"),
    ("Melanie did not move to Berlin.", "move", "polarity", "neg"),
    ("Melanie never moved to Berlin.", "moved", "polarity", "neg"),
    ("Melanie has no interest in moving.", "interest", "polarity", "pos"),
    # --- modality ---
    ("Melanie can play the piano.", "play", "modality", "ability"),
    ("Melanie might move to Berlin.", "move", "modality", "possibility"),
    ("Melanie may move to Berlin.", "move", "modality", "possibility"),
    ("Melanie could move to Berlin.", "move", "modality", "possibility"),
    ("Melanie must move to Berlin.", "move", "modality", "obligation"),
    ("Melanie should move to Berlin.", "move", "modality", "obligation"),
    ("Melanie will move to Berlin.", "move", "modality", "intention"),
    ("Melanie moved to Berlin.", "moved", "modality", "none"),
    # --- voice ---
    ("Melanie sold the piano.", "sold", "voice", "active"),
    ("The piano was sold by Melanie.", "sold", "voice", "passive"),
    # --- epistemic ---
    ("Melanie moved to Berlin.", "moved", "epistemic", "asserted"),
    ("If Melanie moves to Berlin, we will visit.", "moves", "epistemic", "conditional"),
    ("Unless Melanie moves to Berlin, we stay.", "moves", "epistemic", "conditional"),
    ("Caroline said Melanie moved to Berlin.", "moved", "epistemic", "reported"),
    ("Caroline told me Melanie moved to Berlin.", "moved", "epistemic", "reported"),
    ("Melanie wants to move to Berlin.", "move", "epistemic", "desired"),
    ("Melanie plans to move to Berlin.", "move", "epistemic", "desired"),
    ("Melanie hopes to move to Berlin.", "move", "epistemic", "desired"),
    # --- hedging ---
    ("Melanie moved to Berlin.", "moved", "hedged", False),
    ("Melanie probably moved to Berlin.", "moved", "hedged", True),
    ("Maybe Melanie moved to Berlin.", "moved", "hedged", True),
    ("I think Melanie moved to Berlin.", "moved", "hedged", True),
    # --- tense ---
    ("Melanie moved to Berlin.", "moved", "tense", "past"),
    ("Melanie moves to Berlin.", "moves", "tense", "pres"),
]


@pytest.mark.parametrize("text,trigger,field,expected", CASES)
def test_rule(nlp, text: str, trigger: str, field: str, expected) -> None:
    got = getattr(_run(nlp, text, trigger), field)
    assert got == expected, f"{text!r} -> {field}={got!r}, expected {expected!r}"


def test_the_headline_case(nlp) -> None:
    """ "She said she might move" must not be stored as "she moved"."""
    m = _run(nlp, "Caroline said she might move to Berlin.", "move")
    assert m.epistemic == "reported"
    assert m.modality == "possibility"
    assert m.attributed_to == "Caroline"
    assert m.polarity == "pos"


def test_negated_report_keeps_both_signals(nlp) -> None:
    m = _run(nlp, "Caroline said she did not move to Berlin.", "move")
    assert m.epistemic == "reported"
    assert m.polarity == "neg"


def test_no_trigger_degrades_to_defaults_not_to_a_guess(nlp) -> None:
    """The non-SRL path has no span. It must abstain, not analyse a random token."""
    m = analyse(nlp("Melanie did not move to Berlin."), -1)
    assert m.rules == ("no-trigger",)
    assert m.polarity == "pos" and m.epistemic == "asserted"


def test_every_result_carries_its_provenance(nlp) -> None:
    """R4: a heuristic presented as a graph property must be auditable."""
    m = _run(nlp, "Caroline said Melanie might not move to Berlin.", "move")
    assert m.rules, "no rule ids recorded"
    assert "modality_rules" in m.as_props()
    assert m.as_props()["modality_rules"]


def test_mislemmatised_desire_verb_still_fires(nlp) -> None:
    """en_core_web_lg lemmatizes "hopes" -> "hop". A lemma-only lookup misses it,
    and the sentence is then stored as an assertion that she moved."""
    doc = nlp("Melanie hopes to move to Berlin.")
    assert doc[1].lemma_ == "hop", "spaCy fixed the lemma; the surface fallback can go"
    assert _run(nlp, "Melanie hopes to move to Berlin.", "move").epistemic == "desired"
