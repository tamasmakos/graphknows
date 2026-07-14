"""Guard tests for subject-centric SVO relation extraction (svo.py).

Uses the real spaCy ``en_core_web_lg`` pipeline (as the ingest path does) so the
dependency parse, NER types and WordNet supersenses that drive the drop rules
are exercised end-to-end.
"""

from __future__ import annotations

import pytest
import spacy

from graphknows.ingestion.extraction.relations.svo import extract_svo, parse_speaker


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")


def _triples(nlp, text, speaker, speakers=None):
    speakers = speakers if speakers is not None else ([speaker] if speaker else [])
    out = []
    for sent in nlp(text).sents:
        out.extend(extract_svo(sent, speaker, speakers))
    return out


def test_first_person_subject_resolves_to_speaker(nlp):
    triples = _triples(nlp, "I work as a nurse.", "Caroline")
    assert triples, "expected an SVO triple"
    t = triples[0]
    # Subject "I" resolves to the utterance speaker, not a text token.
    assert t["head"] == "Caroline"
    assert t["tail"] == "nurse"
    assert t["head_named"] is True


def test_predicate_is_verb_plus_prep(nlp):
    # The label is the raw verb lemma + preposition ("work" + "as" -> WORK_AS).
    triples = _triples(nlp, "I work as a nurse.", "Caroline")
    assert triples[0]["relation"] == "WORK_AS"


def test_phatic_verb_dropped(nlp):
    # "believe" is a WordNet verb.cognition supersense — a mental/phatic act,
    # not a fact — so it yields no triple.
    assert _triples(nlp, "I believe the story.", "Caroline") == []


def test_date_object_dropped(nlp):
    # A DATE object is temporal metadata, not a relation tail.
    triples = _triples(nlp, "I graduated in 2010.", "Caroline")
    assert all("2010" not in t["tail"] for t in triples)
    assert triples == []


def test_dangling_pronoun_subject_dropped(nlp):
    # An unresolved anaphoric subject ("they") cannot anchor a fact.
    assert _triples(nlp, "They arrived at the party.", "Caroline") == []


def test_named_object_flagged(nlp):
    # A proper-noun / NER object is marked tail_named (gates structural keeps).
    triples = _triples(nlp, "I visited Rome with my brother David.", "Gina")
    rome = next(t for t in triples if t["tail"] == "Rome")
    assert rome["relation"] == "VISIT" and rome["tail_named"] is True
    david = next(t for t in triples if t["tail"] == "David")
    assert david["relation"] == "VISIT_WITH" and david["tail_named"] is True


def test_parse_speaker():
    assert parse_speaker("8:30 pm on 12 May, 2023 | Jon: I lost my job.") == "Jon"
    assert parse_speaker("Caroline: hi there") == "Caroline"
    assert parse_speaker("a line with no speaker prefix") == ""
