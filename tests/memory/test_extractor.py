"""Unit tests for the dynamic spaCy-hinted GLiNER2 extractor (SVO-primary)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import (
    ExtractionResult,
    GLiNER2EntityExtractor,
    _SpacySchemaMiner,
)


def test_extraction_result_has_entities_and_relations_fields():
    result = ExtractionResult(
        entities=[{"name": "OpenAI", "type": "ORG", "score": 0.9}],
        relations=[
            {
                "head": "OpenAI",
                "head_type": "ORG",
                "relation": "BUILD",
                "tail": "ChatGPT",
                "tail_type": "PRODUCT",
                "score": 1.0,
            }
        ],
    )

    assert result.entities[0]["name"] == "OpenAI"
    assert result.relations[0]["relation"] == "BUILD"


def test_spacy_schema_miner_mines_entity_and_svo_schema():
    """Real spaCy: entity schema from NER, relation schema + triples from SVO."""
    miner = _SpacySchemaMiner()

    schema = miner.build("Caroline: I work as a nurse.")

    assert "person" in schema.entity_spec
    assert schema.entity_spec["person"]["threshold"] == pytest.approx(0.35)
    assert schema.label_to_type["person"] == "PERSON"
    # Open-vocab predicate: the label is the verb + preposition the text stated.
    assert "WORK_AS" in schema.relation_spec
    # The first-person subject resolved to the utterance speaker.
    triples = {(t["head"], t["relation"], t["tail"]) for t in schema.svo_triples}
    assert ("Caroline", "WORK_AS", "nurse") in triples


def _mock_extractor(svo_triples, raw_relations, raw_entities=None):
    """Extractor with a mocked GLiNER2 model and a hand-built dynamic schema."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "dynamic person", "threshold": 0.35}},
        relation_spec={"WORK_AS": "dynamic relation"},
        label_to_type={"person": "PERSON"},
        svo_triples=svo_triples,
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema

    mock_schema = MagicMock()
    mock_model = MagicMock()
    mock_model.create_schema.return_value = mock_schema
    mock_model.extract.return_value = {
        "entities": (
            raw_entities
            if raw_entities is not None
            else {"person": [{"text": "Caroline", "confidence": 0.95}]}
        ),
        "relation_extraction": raw_relations,
    }
    extractor = GLiNER2EntityExtractor(
        model_loader=lambda: mock_model,
        schema_miner=schema_miner,
    )
    return extractor, schema, mock_schema, mock_model


def _svo(head, relation, tail, *, head_named, tail_named, verb="work", prep="as"):
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "verb": verb,
        "prep": prep,
        "head_named": head_named,
        "tail_named": tail_named,
    }


def test_svo_triple_promoted_when_gliner_corroborates_object():
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=False)]
    raw = {
        "WORK_AS": [
            {
                "head": {"text": "Caroline", "confidence": 0.8},
                "tail": {"text": "nurse", "confidence": 0.7},
            }
        ]
    }
    extractor, schema, mock_schema, _mm = _mock_extractor(svo, raw)

    result = extractor.extract("Caroline: I work as a nurse.")

    # The mined SVO predicate vocabulary IS the GLiNER2 relation schema.
    mock_schema.relations.assert_called_once_with(schema.relation_spec)
    r = result.relations[0]
    assert (r["head"], r["relation"], r["tail"]) == ("Caroline", "WORK_AS", "nurse")
    assert r["source"] == "svo+gliner"
    assert r["score"] == pytest.approx(0.7)  # min(head, tail) GLiNER2 confidence
    assert r["head_type"] == "PERSON"  # from the entity list
    assert r["tail_type"] == "ENTITY"  # common noun, not an extracted entity


def test_structural_svo_kept_when_both_named_without_corroboration():
    svo = [_svo("Gina", "VISIT", "Rome", head_named=True, tail_named=True, verb="visit", prep="")]
    extractor, *_ = _mock_extractor(svo, {})  # GLiNER2 emits no relations

    result = extractor.extract("Gina: I visited Rome.")

    r = result.relations[0]
    assert (r["relation"], r["tail"]) == ("VISIT", "Rome")
    assert r["source"] == "svo"
    assert r["score"] == pytest.approx(0.6)


def test_uncorroborated_svo_with_common_noun_object_dropped():
    """A tail that is a bare common noun needs GLiNER2 corroboration to survive."""
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=False)]
    extractor, *_ = _mock_extractor(svo, {})

    result = extractor.extract("Caroline: I work as a nurse.")

    assert result.relations == []


def test_subthreshold_gliner_confidence_is_not_corroboration():
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=False)]
    raw = {
        "WORK_AS": [
            {
                "head": {"text": "Caroline", "confidence": 0.9},
                "tail": {"text": "nurse", "confidence": 0.2},  # min 0.2 < 0.35
            }
        ]
    }
    extractor, *_ = _mock_extractor(svo, raw)

    result = extractor.extract("Caroline: I work as a nurse.")

    assert result.relations == []


def test_gliner_tail_match_is_lenient_substring():
    """GLiNER2's object surface may be longer than the SVO head-noun ('a nurse')."""
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=False)]
    raw = {
        "WORK_AS": [
            {
                "head": {"text": "Caroline", "confidence": 0.8},
                "tail": {"text": "a nurse", "confidence": 0.6},
            }
        ]
    }
    extractor, *_ = _mock_extractor(svo, raw)

    result = extractor.extract("Caroline: I work as a nurse.")

    assert result.relations[0]["source"] == "svo+gliner"


def test_subject_agreement_rescues_coreferent_object():
    """GLiNER2 names the object differently ('Luna') but agrees on the subject."""
    svo = [_svo("Gina", "ADOPT", "cat", head_named=True, tail_named=False, verb="adopt", prep="")]
    raw = {
        "ADOPT": [
            {
                "head": {"text": "Gina", "confidence": 0.9},
                "tail": {"text": "Luna", "confidence": 0.9},  # coreferent, != "cat"
            }
        ]
    }
    extractor, *_ = _mock_extractor(svo, raw)

    result = extractor.extract("Gina: I adopted a cat named Luna.")

    r = result.relations[0]
    assert (r["relation"], r["tail"]) == ("ADOPT", "cat")  # SVO's cleaner object
    assert r["source"] == "svo+gliner"


def test_explicit_relation_schema_overrides_mined_svo_schema():
    """An explicitly passed taxonomy schema (escape hatch) is handed to GLiNER2."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "p", "threshold": 0.35}},
        relation_spec={"WORK_AS": "desc"},
        label_to_type={"person": "PERSON"},
        svo_triples=[],
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_schema = MagicMock()
    mock_model = MagicMock()
    mock_model.create_schema.return_value = mock_schema
    mock_model.extract.return_value = {"entities": {}, "relation_extraction": {}}
    taxonomy_schema = {"created": "one entity created another"}
    extractor = GLiNER2EntityExtractor(
        model_loader=lambda: mock_model,
        schema_miner=schema_miner,
        relation_schema=taxonomy_schema,
    )

    extractor.extract("OpenAI built ChatGPT.")

    mock_schema.relations.assert_called_once_with(taxonomy_schema)
    assert extractor._relation_schema == taxonomy_schema


def test_extract_returns_empty_when_spacy_mines_no_entity_schema():
    schema = SimpleNamespace(
        entity_spec={}, relation_spec={}, label_to_type={}, svo_triples=[]
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    result = extractor.extract("No named entities here.")

    assert result == ExtractionResult()
    mock_model.extract.assert_not_called()


def test_extract_filters_low_confidence_and_blank_entities():
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "dynamic person", "threshold": 0.35}},
        relation_spec={},
        label_to_type={"person": "PERSON"},
        svo_triples=[],
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    mock_model.create_schema.return_value = MagicMock()
    mock_model.extract.return_value = {
        "entities": {
            "person": [
                {"text": "Alice", "confidence": 0.9},
                {"text": "Bob", "confidence": 0.2},
                {"text": " ", "confidence": 0.9},
            ]
        },
        "relation_extraction": {},
    }
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    result = extractor.extract("Alice met Bob.")

    assert result.entities == [{"name": "Alice", "type": "PERSON", "score": 0.9}]


def test_ensure_model_loaded_raises_if_model_loader_returns_none():
    extractor = GLiNER2EntityExtractor(model_loader=lambda: None, schema_miner=MagicMock())

    with pytest.raises(RuntimeError, match="GLiNER2 model failed to load"):
        extractor._ensure_model_loaded()
