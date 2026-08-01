"""The relation label space is the ontology's, end to end through the real model.

``extract(text, extra_relation_labels=...)`` is handed ``{property label ->
definition}`` for the ontology properties nearest the chunk embedding
(``stm/ingest.py::_label_hints``). Those labels ARE the relation spec — since the
SVO miner was deleted there is no other source — so a relation that comes out
carries the ontology property's own name.

These run the real relex model, because the claim being tested is that relex
answers to a schema.org label at all. A mocked model can only prove the label
reached it.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.extraction.entities.extractor import GLiNER2EntityExtractor

pytestmark = pytest.mark.integration

# A slice of the bundled schema.org digest, in the shape _label_hints emits.
_SCHEMA_ORG = {
    "knows": "The most generic bi-directional social/work relation.",
    "worksFor": "Organizations that the person works for.",
    "spouse": "The person the subject is married to.",
}


@pytest.fixture(scope="module")
def extractor() -> GLiNER2EntityExtractor:
    return GLiNER2EntityExtractor()


def test_a_schema_org_property_becomes_the_relation_name(
    extractor: GLiNER2EntityExtractor,
) -> None:
    result = extractor.extract(
        "Melanie knows Caroline from college.", extra_relation_labels=dict(_SCHEMA_ORG)
    )

    assert [(r["head"], r["relation"], r["tail"]) for r in result.relations] == [
        ("Melanie", "knows", "Caroline")
    ]
    assert result.relations[0]["source"] == "relex"


def test_no_possessive_has_relation_is_produced(extractor: GLiNER2EntityExtractor) -> None:
    """``HAS`` was 25 of SVO's 28 edges on conv-30 and is not a schema.org property.

    ``Jon HAS back`` / ``Jon HAS corner`` / ``Jon HAS side`` all came from the
    possessive pattern on exactly this shape of sentence. Nothing may mint a
    relation the ontology did not name.
    """
    result = extractor.extract(
        "Jon: My back hurts, and Caroline's studio is next to my corner.",
        extra_relation_labels=dict(_SCHEMA_ORG),
    )

    labels = {r["relation"] for r in result.relations}
    assert "HAS" not in labels
    assert labels <= set(_SCHEMA_ORG), "only an ontology property may name an edge"


def test_relations_are_confined_to_the_supplied_property_labels(
    extractor: GLiNER2EntityExtractor,
) -> None:
    """No open-vocabulary predicate can appear: relex may only emit a label it was given."""
    result = extractor.extract(
        "Melanie knows Caroline. Melanie works for Mercy Hospital and visited Rome with David.",
        extra_relation_labels=dict(_SCHEMA_ORG),
    )

    assert {r["relation"] for r in result.relations} <= set(_SCHEMA_ORG)


def test_observed_ner_label_does_not_suppress_an_ontology_relation(
    extractor: GLiNER2EntityExtractor,
) -> None:
    """This was a strict xfail blamed on an entity-label collision. It was the THRESHOLD.

    The recorded diagnosis was that a spaCy-observed label near-duplicating a
    base-inventory one ('org' beside 'organization') cost relex the relation.
    What actually happened is that the observed label moved the score, and
    _RELATION_THRESHOLD was 0.7 — the model-card figure, calibrated for the
    model's own label vocabulary, not for injected ontology property names. The
    relation is scored 0.437 with 'org' present. Lowering the floor to 0.3 makes
    it appear, so the xfail is removed rather than re-explained.
    """
    result = extractor.extract(
        "Caroline works for Mercy Hospital as a nurse.", extra_relation_labels=dict(_SCHEMA_ORG)
    )

    assert [(r["head"], r["relation"], r["tail"]) for r in result.relations] == [
        ("Caroline", "worksFor", "Mercy Hospital")
    ]
