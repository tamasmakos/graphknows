"""Unit tests for GLiNER2EntityExtractor ExtractionResult."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphknows.memory.extractor import ExtractionResult, GLiNER2EntityExtractor


def test_extraction_result_has_entities_and_relations_fields():
    result = ExtractionResult(
        entities=[{"name": "EU", "type": "ORG", "score": 0.9}],
        relations=[{"head": "EU", "head_type": "ORG", "relation": "REGULATES", "tail": "ECB", "tail_type": "ORG", "score": 1.0}],
    )
    assert len(result.entities) == 1
    assert result.entities[0]["name"] == "EU"
    assert len(result.relations) == 1
    assert result.relations[0]["relation"] == "REGULATES"


def test_extract_returns_extraction_result_with_entities_and_relations():
    mock_model = MagicMock()
    mock_model.extract.return_value = {
        "entities": {
            "PERSON": [{"text": "Viktor Orbán", "label": "PERSON", "confidence": 0.95}],
        },
        "relation_extraction": {
            "OPPOSES": [
                {
                    "head": {"text": "Viktor Orbán", "confidence": 1.0},
                    "tail": {"text": "Ukraine EU membership", "confidence": 1.0},
                }
            ]
        },
    }

    with (
        patch.object(GLiNER2EntityExtractor, "_ensure_model_loaded"),
        patch.object(GLiNER2EntityExtractor, "_child_to_group", {"person": "PERSON"}, create=True),
        patch.object(GLiNER2EntityExtractor, "_taxonomy", create=True) as mock_taxonomy,
    ):
        mock_taxonomy.get_min_ner_confidence.return_value = 0.0
        mock_taxonomy.entity_groups = {}
        extractor = GLiNER2EntityExtractor.__new__(GLiNER2EntityExtractor)
        extractor._gliner = mock_model
        extractor._gliner_schema = MagicMock()
        result = extractor.extract("Viktor Orbán opposes Ukraine EU membership.")

    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 1
    assert result.entities[0]["name"] == "Viktor Orbán"
    assert result.entities[0]["type"] == "PERSON"
    assert len(result.relations) == 1
    assert result.relations[0]["head"] == "Viktor Orbán"
    assert result.relations[0]["relation"] == "OPPOSES"
    assert result.relations[0]["tail"] == "Ukraine EU membership"
