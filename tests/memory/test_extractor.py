"""Unit tests for GLiNER2EntityExtractor and ExtractionResult."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphknows.memory.extractor import ExtractionResult, GLiNER2EntityExtractor

# ---------------------------------------------------------------------------
# Standalone tests (test_extractor.py)
# ---------------------------------------------------------------------------


def test_extraction_result_has_entities_and_relations_fields():
    result = ExtractionResult(
        entities=[{"name": "EU", "type": "ORG", "score": 0.9}],
        relations=[
            {
                "head": "EU",
                "head_type": "ORG",
                "relation": "REGULATES",
                "tail": "ECB",
                "tail_type": "ORG",
                "score": 1.0,
            }
        ],
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


# ---------------------------------------------------------------------------
# Class-based tests (test_wave3_extractor.py)
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_defaults_empty_lists(self):
        r = ExtractionResult()
        assert r.entities == []
        assert r.relations == []

    def test_accepts_data(self):
        r = ExtractionResult(
            entities=[{"name": "Alice", "type": "PERSON", "score": 0.9}],
            relations=[{"head": "Alice", "relation": "KNOWS", "tail": "Bob", "score": 0.8}],
        )
        assert len(r.entities) == 1
        assert len(r.relations) == 1


def _make_extractor():
    """Create GLiNER2EntityExtractor with a mocked taxonomy (no file I/O)."""
    # Build a minimal mock taxonomy
    mock_taxonomy = MagicMock()
    mock_taxonomy.name = "test"
    mock_taxonomy.relation_types = ["KNOWS", "WORKS_FOR"]
    mock_group_person = MagicMock()
    mock_group_person.min_ner_confidence = 0.5
    mock_group_person.description = "A person"
    mock_group_person.children = []
    mock_taxonomy.entity_groups = {"PERSON": mock_group_person}
    mock_taxonomy.get_min_ner_confidence.return_value = 0.35

    with patch(
        "graphknows.memory.extractor.GLiNER2EntityExtractor.__init__",
        lambda self, taxonomy_path=None: None,
    ):
        extractor = GLiNER2EntityExtractor.__new__(GLiNER2EntityExtractor)

    extractor._taxonomy = mock_taxonomy
    extractor._gliner = None
    extractor._gliner_schema = None
    extractor._child_to_group = {"person": "PERSON"}
    extractor._entity_spec = {"person": {"description": "A person", "threshold": 0.5}}
    extractor._rel_descs = {"KNOWS": "knows", "WORKS_FOR": "works for"}
    return extractor


class TestGLiNER2EntityExtractorInit:
    def test_builds_entity_spec_from_taxonomy(self):
        """Ensure entity_spec excludes groups with empty descriptions."""
        mock_taxonomy = MagicMock()
        mock_taxonomy.name = "test"
        mock_taxonomy.relation_types = []
        grp_has_desc = MagicMock()
        grp_has_desc.min_ner_confidence = 0.5
        grp_has_desc.description = "  A person  "
        grp_has_desc.children = []
        grp_no_desc = MagicMock()
        grp_no_desc.min_ner_confidence = 0.5
        grp_no_desc.description = ""  # empty — should be excluded
        grp_no_desc.children = []
        mock_taxonomy.entity_groups = {"PERSON": grp_has_desc, "NOTYPE": grp_no_desc}

        with patch(
            "graphknows.graphgen.config.taxonomy.get_default_taxonomy",
            return_value=mock_taxonomy,
        ):
            extractor = GLiNER2EntityExtractor()

        assert "person" in extractor._entity_spec
        assert "notype" not in extractor._entity_spec

    def test_rel_descs_built_from_taxonomy(self):
        mock_taxonomy = MagicMock()
        mock_taxonomy.name = "t"
        mock_taxonomy.relation_types = ["KNOWS_OF", "WORKS_AT"]
        mock_taxonomy.entity_groups = {}

        with patch(
            "graphknows.graphgen.config.taxonomy.get_default_taxonomy",
            return_value=mock_taxonomy,
        ):
            extractor = GLiNER2EntityExtractor()

        assert "KNOWS_OF" in extractor._rel_descs
        assert extractor._rel_descs["KNOWS_OF"] == "knows of"


class TestEnsureModelLoaded:
    def test_no_op_when_already_loaded(self):
        extractor = _make_extractor()
        mock_model = MagicMock()
        extractor._gliner = mock_model
        extractor._ensure_model_loaded()
        assert extractor._gliner is mock_model

    def test_raises_if_model_returns_none(self):
        extractor = _make_extractor()
        extractor._gliner = None

        with patch(
            "graphknows.graphgen.pipeline.entity_relation.extraction.get_gliner2_model",
            return_value=None,
        ), pytest.raises(RuntimeError, match="GLiNER2 model failed to load"):
            extractor._ensure_model_loaded()

    def test_loads_model_and_sets_attrs(self):
        extractor = _make_extractor()
        extractor._gliner = None

        mock_model = MagicMock()
        mock_schema = MagicMock()
        mock_model.create_schema.return_value = mock_schema

        with patch(
            "graphknows.graphgen.pipeline.entity_relation.extraction.get_gliner2_model",
            return_value=mock_model,
        ):
            extractor._ensure_model_loaded()

        assert extractor._gliner is mock_model
        assert extractor._gliner_schema is mock_schema
        mock_schema.entities.assert_called_once()


class TestExtract:
    def _make_ready(self, raw_output: dict):
        """Extractor with model already loaded returning raw_output."""
        extractor = _make_extractor()
        mock_model = MagicMock()
        mock_model.extract.return_value = raw_output
        extractor._gliner = mock_model
        extractor._gliner_schema = MagicMock()
        return extractor

    def test_empty_raw_returns_empty_result(self):
        extractor = self._make_ready({"entities": {}, "relation_extraction": {}})
        result = extractor.extract("some text")
        assert result.entities == []
        assert result.relations == []

    def test_parses_entities_above_threshold(self):
        raw = {
            "entities": {
                "person": [
                    {"text": "Alice", "confidence": 0.9},
                    {"text": "Bob", "confidence": 0.2},  # below _BASE_THRESHOLD 0.35
                ]
            },
            "relation_extraction": {},
        }
        extractor = self._make_ready(raw)
        extractor._child_to_group = {"person": "PERSON"}
        extractor._taxonomy.get_min_ner_confidence.return_value = 0.35

        result = extractor.extract("Alice meets Bob")
        names = [e["name"] for e in result.entities]
        assert "Alice" in names
        assert "Bob" not in names

    def test_skips_empty_entity_text(self):
        raw = {
            "entities": {
                "person": [
                    {"text": "  ", "confidence": 0.9},  # blank text
                    {"text": "Alice", "confidence": 0.8},
                ]
            },
            "relation_extraction": {},
        }
        extractor = self._make_ready(raw)
        extractor._taxonomy.get_min_ner_confidence.return_value = 0.35

        result = extractor.extract("Alice")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Alice"

    def test_parses_relations(self):
        raw = {
            "entities": {},
            "relation_extraction": {
                "KNOWS": [
                    {
                        "head": {"text": "Alice", "confidence": 0.85},
                        "tail": {"text": "Bob", "confidence": 0.80},
                    }
                ]
            },
        }
        extractor = self._make_ready(raw)
        result = extractor.extract("Alice knows Bob")
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel["head"] == "Alice"
        assert rel["tail"] == "Bob"
        assert rel["relation"] == "KNOWS"

    def test_skips_relations_with_empty_head_or_tail(self):
        raw = {
            "entities": {},
            "relation_extraction": {
                "KNOWS": [
                    {
                        "head": {"text": "", "confidence": 0.9},
                        "tail": {"text": "Bob", "confidence": 0.9},
                    },
                ]
            },
        }
        extractor = self._make_ready(raw)
        result = extractor.extract("test")
        assert result.relations == []

    def test_relation_score_is_minimum_of_head_tail(self):
        raw = {
            "entities": {},
            "relation_extraction": {
                "KNOWS": [
                    {
                        "head": {"text": "Alice", "confidence": 0.9},
                        "tail": {"text": "Bob", "confidence": 0.6},
                    }
                ]
            },
        }
        extractor = self._make_ready(raw)
        result = extractor.extract("test")
        assert result.relations[0]["score"] == pytest.approx(0.6)

    def test_entity_type_canonical_mapped_via_child_to_group(self):
        raw = {
            "entities": {"politician": [{"text": "Macron", "confidence": 0.88}]},
            "relation_extraction": {},
        }
        extractor = self._make_ready(raw)
        extractor._child_to_group = {"politician": "PERSON"}  # child→canonical
        extractor._taxonomy.get_min_ner_confidence.return_value = 0.35

        result = extractor.extract("Macron is a politician")
        assert result.entities[0]["type"] == "PERSON"
