"""Overlapping-span resolution — Rule "Typed entities are extracted against the
candidate ontology classes":

    Scenario: Overlapping spans are resolved to a single entity
      Given the extractor returns "Ada Lovelace" typed "Person" with confidence 0.9
      And the extractor returns "Lovelace" typed "Person" with confidence 0.7
      And the two spans overlap
      When overlapping spans are resolved
      Then only the higher-confidence, longer span is retained
      And the discarded span is recorded as an alias candidate for the retained entity

relex runs with ``flat_ner=False``, so it returns nested spans by design: on the
real chunk below it emits 'new', 'new offers' AND 'offers' as three PRODUCT
entities for one thing. Every span it returns carries ``start``/``end`` character
offsets, so overlap is resolved on the offsets themselves.
"""

from __future__ import annotations

from graphknows.ingestion.extraction.entities.extractor import _dedupe_entities

_PERSON = {"person": "PERSON"}
_PRODUCT = {"person": "PERSON", "product": "PRODUCT"}


def _by_name(entities: list[dict]) -> dict[str, dict]:
    return {e["name"]: e for e in entities}


def test_overlapping_spans_resolve_to_the_longer_higher_confidence_span():
    """The spec scenario, verbatim."""
    spans = [
        {"start": 0, "end": 12, "text": "Ada Lovelace", "label": "person", "score": 0.9},
        {"start": 4, "end": 12, "text": "Lovelace", "label": "person", "score": 0.7},
    ]

    entities = _dedupe_entities(spans, _PERSON)

    assert [e["name"] for e in entities] == ["Ada Lovelace"]
    assert entities[0]["type"] == "PERSON"
    assert entities[0]["score"] == 0.9
    assert entities[0]["aliases"] == ["Lovelace"]


def test_equal_length_overlap_keeps_the_higher_confidence_span():
    """Same span, two labels — the score decides, the loser is an alias."""
    spans = [
        {"start": 0, "end": 8, "text": "Lovelace", "label": "person", "score": 0.4},
        {"start": 0, "end": 8, "text": "lovelace", "label": "person", "score": 0.8},
    ]

    entities = _dedupe_entities(spans, _PERSON)

    assert [e["name"] for e in entities] == ["lovelace"]
    assert entities[0]["score"] == 0.8


def test_prefix_subspans_of_one_noun_phrase_collapse_to_one_entity():
    """The measured regression: relex's own spans for a real chunk.

    "Customers love new offers and promotions." — verbatim relex output. 'new' is
    a bare adjective and 'new'/'offers' are sub-spans of 'new offers'; all three
    became separate ENTITY nodes ('best'/'best moves', 'biz'/'biz plan' in the
    live conv-30 graph came from exactly this).
    """
    spans = [
        {"start": 0, "end": 9, "text": "Customers", "label": "person", "score": 0.99},
        {"start": 15, "end": 18, "text": "new", "label": "product", "score": 0.86},
        {"start": 15, "end": 25, "text": "new offers", "label": "product", "score": 0.77},
        {"start": 19, "end": 25, "text": "offers", "label": "product", "score": 0.91},
        {"start": 30, "end": 40, "text": "promotions", "label": "product", "score": 0.97},
    ]

    entities = _dedupe_entities(spans, _PRODUCT)

    by_name = _by_name(entities)
    assert set(by_name) == {"Customers", "new offers", "promotions"}
    assert sorted(by_name["new offers"]["aliases"]) == ["new", "offers"]


def test_non_overlapping_same_type_entities_are_not_merged():
    """Two people in one chunk stay two people."""
    spans = [
        {"start": 0, "end": 4, "text": "Gina", "label": "person", "score": 0.9},
        {"start": 9, "end": 12, "text": "Jon", "label": "person", "score": 0.9},
    ]

    entities = _dedupe_entities(spans, _PERSON)

    assert sorted(e["name"] for e in entities) == ["Gina", "Jon"]
    assert all("aliases" not in e for e in entities)


def test_a_speaker_glued_span_does_not_swallow_the_name_inside_it():
    """``"Caroline: Thanks, Melanie"`` reduces to "Caroline" — and must then stop
    overlapping "Melanie", which sits in the text it no longer covers."""
    spans = [
        {
            "start": 0,
            "end": 25,
            "text": "Caroline: Thanks, Melanie",
            "label": "person",
            "score": 0.9,
        },
        {"start": 18, "end": 25, "text": "Melanie", "label": "person", "score": 0.8},
    ]

    entities = _dedupe_entities(spans, _PERSON, speakers=frozenset({"caroline"}))

    assert sorted(e["name"] for e in entities) == ["Caroline", "Melanie"]


def test_spans_without_offsets_are_all_retained():
    """Callers that stub relex without offsets get the pre-existing behaviour."""
    spans = [
        {"text": "Ada Lovelace", "label": "person", "score": 0.9},
        {"text": "Lovelace", "label": "person", "score": 0.7},
    ]

    entities = _dedupe_entities(spans, _PERSON)

    assert sorted(e["name"] for e in entities) == ["Ada Lovelace", "Lovelace"]
