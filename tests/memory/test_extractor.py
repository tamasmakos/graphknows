"""Unit tests for the dynamic spaCy-hinted GLiNER2 extractor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult, GLiNER2EntityExtractor, _SpacySchemaMiner


class _Tok:
    def __init__(
        self,
        idx: int,
        text: str,
        lemma: str,
        pos: str,
        ancestors: list[_Tok] | None = None,
    ) -> None:
        self.i = idx
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self._ancestors = ancestors or []

    @property
    def ancestors(self) -> list[_Tok]:
        return self._ancestors


def test_extraction_result_has_entities_and_relations_fields():
    result = ExtractionResult(
        entities=[{"name": "OpenAI", "type": "ORG", "score": 0.9}],
        relations=[
            {
                "head": "OpenAI",
                "head_type": "ORG",
                "relation": "build",
                "tail": "ChatGPT",
                "tail_type": "PRODUCT",
                "score": 1.0,
            }
        ],
    )

    assert result.entities[0]["name"] == "OpenAI"
    assert result.relations[0]["relation"] == "build"


def test_spacy_schema_miner_builds_entity_and_relation_specs_from_doc():
    root = _Tok(1, "built", "build", "VERB")
    left = _Tok(0, "OpenAI", "OpenAI", "PROPN", [root])
    right = _Tok(2, "ChatGPT", "ChatGPT", "PROPN", [root])
    sent = SimpleNamespace(start=0)
    doc = SimpleNamespace(
        ents=[
            SimpleNamespace(text="OpenAI", label_="ORG", root=left, sent=sent),
            SimpleNamespace(text="ChatGPT", label_="PRODUCT", root=right, sent=sent),
        ]
    )
    miner = _SpacySchemaMiner(nlp_loader=lambda _name: lambda _text: doc)

    schema = miner.build("OpenAI built ChatGPT.")

    assert schema.entity_spec["org"]["description"]
    assert schema.entity_spec["product"]["threshold"] == pytest.approx(0.35)
    assert schema.label_to_type == {"org": "ORG", "product": "PRODUCT"}
    assert "build" in schema.relation_spec
    assert "ORG" not in schema.relation_spec["build"]
    assert "OpenAI -> build -> ChatGPT" in schema.relation_spec["build"]
    assert "avoid selecting whole clauses" in schema.relation_spec["build"]


def test_spacy_schema_miner_skips_prepositions_adjectives_and_parent_verbs():
    said = _Tok(1, "said", "say", "VERB")
    support = _Tok(2, "supports", "support", "VERB", [said])
    german = _Tok(4, "German", "german", "ADJ", [support, said])
    in_token = _Tok(3, "in", "in", "ADP", [german, support, said])
    left = _Tok(0, "Commission", "Commission", "PROPN", [said])
    right = _Tok(5, "Green Deal", "Green Deal", "PROPN", [in_token, german, support, said])
    sent = SimpleNamespace(start=0)
    doc = SimpleNamespace(
        ents=[
            SimpleNamespace(text="Commission", label_="ORG", root=left, sent=sent),
            SimpleNamespace(text="Green Deal", label_="PRODUCT", root=right, sent=sent),
        ]
    )
    miner = _SpacySchemaMiner(nlp_loader=lambda _name: lambda _text: doc)

    schema = miner.build("The Commission said it supports the Green Deal.")

    assert "support" in schema.relation_spec
    assert "Commission -> support -> Green Deal" in schema.relation_spec["support"]
    assert "dependency path:" in schema.relation_spec["support"]
    assert "say" not in schema.relation_spec
    assert "in" not in schema.relation_spec
    assert "german" not in schema.relation_spec


def test_extract_builds_fresh_schema_and_parses_entities_and_relations():
    schema = SimpleNamespace(
        entity_spec={"org": {"description": "dynamic org", "threshold": 0.35}},
        relation_spec={"build": "dynamic relation"},
        label_to_type={"org": "ORG"},
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    # Object cleaning parses the relation tail with the miner's spaCy pipeline.
    schema_miner.nlp = _SpacySchemaMiner().nlp

    mock_schema = MagicMock()
    mock_model = MagicMock()
    mock_model.create_schema.return_value = mock_schema
    mock_model.extract.return_value = {
        "entities": {"org": [{"text": "OpenAI", "confidence": 0.95}]},
        "relation_extraction": {
            "build": [
                {
                    "head": {"text": "OpenAI", "confidence": 0.9},
                    "tail": {"text": "ChatGPT", "confidence": 0.8},
                }
            ]
        },
    }
    extractor = GLiNER2EntityExtractor(
        model_loader=lambda: mock_model,
        schema_miner=schema_miner,
    )

    result = extractor.extract("OpenAI built ChatGPT.")

    mock_schema.entities.assert_called_once_with(schema.entity_spec)
    # Entities stay dynamic (spaCy-mined), but RELATIONS come from the configured
    # taxonomy's canonical predicate schema, NOT the per-chunk mined relation_spec.
    mock_schema.relations.assert_called_once_with(extractor._relation_schema)
    assert "build" not in extractor._relation_schema  # mined lemma not used
    mock_model.extract.assert_called_once()
    assert result.entities == [{"name": "OpenAI", "type": "ORG", "score": 0.95}]
    assert result.relations[0]["head"] == "OpenAI"
    assert result.relations[0]["tail"] == "ChatGPT"
    assert result.relations[0]["score"] == pytest.approx(0.8)


def test_extract_returns_empty_when_spacy_mines_no_entity_schema():
    schema = SimpleNamespace(entity_spec={}, relation_spec={}, label_to_type={})
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
