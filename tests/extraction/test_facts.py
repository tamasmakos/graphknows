"""Guard tests for the enhanced deterministic fact extractor (facts.py).

Exercise the levers that break the bare-SVO coverage ceiling: first-person-plural
resolution, xcomp activities, conjunction expansion, head-noun trimming, and the
negation quality guard. Uses the real spaCy model (as the ingest path does).
"""

from __future__ import annotations

import pytest

from graphknows.extraction.relations.facts import extract_facts

_SPEAKERS = [{"name": "Melanie"}, {"name": "Caroline"}]


def _triples(text, **kw):
    facts = extract_facts(text, _SPEAKERS, use_patterns=True, **kw)
    return {(f["source"].lower(), f["relation"], f["target"].lower()) for f in facts}


def _objects_for(text, subject, **kw):
    return {
        f["target"].lower()
        for f in extract_facts(text, _SPEAKERS, use_patterns=True, **kw)
        if f["source"].lower() == subject
    }


def test_first_person_plural_resolves_to_speaker():
    # "We went to the museum" said by Melanie -> the fact is attributed to Melanie.
    objs = _objects_for("Melanie: We went to the museum on Sunday.", "melanie")
    assert any("museum" in o for o in objs)


def test_conjunction_expansion_splits_objects():
    objs = _objects_for("Caroline: I really enjoy camping and hiking.", "caroline")
    assert any("camping" in o for o in objs) and any("hiking" in o for o in objs)


def test_xcomp_activity_is_captured():
    # "went hiking" — the activity is an xcomp verb, not a direct object.
    objs = _objects_for("Melanie: Last weekend I went hiking with the kids.", "melanie")
    assert any("hik" in o for o in objs)


def test_negation_is_dropped():
    # A negated clause must not be emitted as a positive fact.
    trips = _triples("Caroline: I did not go to the parade this year.")
    assert not any(rel == "WENT_TO" and "parade" in obj for _, rel, obj in trips)


def test_head_noun_trim_collapses_clause_object():
    # The clause-y object should collapse toward a head noun, not leak the whole clause.
    objs = _objects_for(
        "Caroline: I have all the support from my friends and mentors.", "caroline"
    )
    assert objs, "expected at least one fact"
    assert min(len(o.split()) for o in objs) <= 3


def test_returns_svo_triple_shape():
    facts = extract_facts("Melanie: I made a clay bowl.", _SPEAKERS, use_patterns=True)
    assert facts and {"source", "relation", "target", "evidence", "confidence"} <= set(facts[0])
