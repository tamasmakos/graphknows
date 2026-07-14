"""Guard tests for subject-centric SVO relation extraction (svo.py).

Uses the real spaCy ``en_core_web_lg`` pipeline (as the ingest path does) so the
dependency parse, NER types and WordNet supersenses that drive the drop rules
are exercised end-to-end.
"""

from __future__ import annotations

import pytest
import spacy

from graphknows.ingestion.extraction.relations.svo import (
    extract_relations,
    extract_svo,
    parse_speaker,
)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")


def _triples(nlp, text, speaker, speakers=None):
    speakers = speakers if speakers is not None else ([speaker] if speaker else [])
    out = []
    for sent in nlp(text).sents:
        out.extend(extract_svo(sent, speaker, speakers))
    return out


def _rels(nlp, text, speaker, speakers=None):
    speakers = speakers if speakers is not None else ([speaker] if speaker else [])
    out = []
    for sent in nlp(text).sents:
        out.extend(extract_relations(sent, speaker, speakers))
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


# --- Rich dependency-grammar relations (D1) --------------------------------


def _rel_set(rels):
    return {(r["head"], r["relation"], r["tail"]) for r in rels}


def test_copular_nominal_is_a(nlp):
    # "I am a nurse" -> IS_A with the subject resolved to the speaker.
    rels = _rels(nlp, "I am a nurse.", "Caroline")
    r = next(r for r in rels if r["relation"] == "IS_A")
    assert (r["head"], r["tail"]) == ("Caroline", "nurse")
    assert r["pattern"] == "copular" and r["head_named"] is True


def test_copular_identity_phrase_survives_as_nominal(nlp):
    # An amod'd nominal head ("transgender woman") is kept — it's an identity,
    # not a bare predicate adjective.
    rels = _rels(nlp, "I am a transgender woman.", "Jon")
    assert ("Jon", "IS_A", "transgender woman") in _rel_set(rels)


def test_copular_predicate_adjective_dropped(nlp):
    # "is excited" is a transient state (acomp adjective), not a durable IS_A.
    rels = _rels(nlp, "I am excited.", "Caroline")
    assert not [r for r in rels if r["relation"] == "IS_A"]


def test_copular_abstract_nominal_dropped(nlp):
    # An abstract nominal predicate (WordNet noun.feeling/cognition) is not an
    # identity — dropped by the concrete-common-noun IS_A gate.
    assert not [r for r in _rels(nlp, "It is my passion.", "Jon") if r["relation"] == "IS_A"]


def test_copular_named_predicate_dropped(nlp):
    # Identity is not another proper noun / capitalized interjection: a NAMED
    # predicate is dropped (kills "Jon IS_A Gina" / "Jon IS_A Thanks" dialogue noise).
    rels = _rels(nlp, "Melanie, thanks! You are Gina.", "Melanie", ["Gina", "Melanie"])
    assert not [r for r in rels if r["relation"] == "IS_A" and r["tail"] == "Gina"]


def test_possessive_has_concrete_tail_kept(nlp):
    # A concrete possessed noun (WordNet noun.artifact) is kept.
    rels = _rels(nlp, "That is Caroline's studio.", "")
    r = next(r for r in rels if r["relation"] == "HAS")
    assert (r["head"], r["tail"]) == ("Caroline", "studio")
    assert r["pattern"] == "possessive" and r["head_named"] is True


def test_possessive_has_abstract_tail_dropped(nlp):
    # An abstract possessed noun (WordNet noun.cognition/feeling) is dropped.
    assert not [r for r in _rels(nlp, "That is Caroline's memory.", "") if r["relation"] == "HAS"]
    assert not [r for r in _rels(nlp, "That is Caroline's passion.", "") if r["relation"] == "HAS"]


def test_possessive_first_person_resolves_to_speaker(nlp):
    rels = _rels(nlp, "My dog is friendly.", "Caroline")
    assert ("Caroline", "HAS", "dog") in _rel_set(rels)


def test_appositive_is_a_named_head(nlp):
    # "Melanie, a nurse" -> (Melanie, IS_A, nurse), named side oriented as head.
    rels = _rels(nlp, "Melanie, a nurse, loves hiking.", "")
    r = next(r for r in rels if r["pattern"] == "appositive")
    assert (r["head"], r["relation"], r["tail"]) == ("Melanie", "IS_A", "nurse")
    assert r["head_named"] is True


def test_noun_anchored_prepositional(nlp):
    # A named head noun with a prep->pobj -> (named, PREP, pobj); pobj not a date.
    rels = _rels(nlp, "I visited the University of Michigan.", "Gina")
    nps = [r for r in rels if r["pattern"] == "noun_prep"]
    assert nps and nps[0]["relation"] == "OF"
    assert nps[0]["head_named"] is True and nps[0]["tail"] == "Michigan"


def test_extract_relations_dedups_and_unions_svo(nlp):
    # SVO facts and rich facts coexist, deduped by (head, relation, tail).
    rels = _rels(nlp, "I am a nurse.", "Caroline")
    keys = [k for k in _rel_set(rels)]
    assert len(keys) == len(set(keys))  # no dup triples
    assert all("pattern" in r for r in rels)
