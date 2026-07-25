"""Tests for the GLiNER2 label-space helpers (schema build, candidate selection).

Formerly also covered the manifest/registry/router ontology stack, which was
removed once it was established to have no production callers.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_taxonomy(entity_groups=None):
    """Create a minimal taxonomy mock."""
    taxonomy = MagicMock()
    taxonomy.entity_groups = entity_groups or {}
    return taxonomy


class TestBuildGliner2Schema:
    def test_returns_entity_dict_without_taxonomy(self):
        from graphknows.ingestion.extraction.entities.label_space import build_gliner2_schema

        labels = ["PERSON", "ORGANIZATION", "LOCATION"]
        result = build_gliner2_schema(labels)
        assert "entities" in result
        assert "person" in result["entities"]
        assert "organization" in result["entities"]

    def test_custom_label_descriptions_override_defaults(self):
        from graphknows.ingestion.extraction.entities.label_space import build_gliner2_schema

        labels = ["PERSON"]
        result = build_gliner2_schema(labels, label_descriptions={"PERSON": "My custom desc"})
        assert result["entities"]["person"] == "My custom desc"

    def test_unknown_label_falls_back_to_lowercase(self):
        from graphknows.ingestion.extraction.entities.label_space import build_gliner2_schema

        labels = ["CUSTOM_TYPE"]
        result = build_gliner2_schema(labels)
        assert result["entities"]["custom_type"] == "custom_type"

    def test_with_taxonomy_merges_descriptions(self):
        from graphknows.ingestion.extraction.entities.label_space import build_gliner2_schema

        group = MagicMock()
        group.description = "A real person."
        taxonomy = _make_taxonomy(entity_groups={"PERSON": group})

        result = build_gliner2_schema(["PERSON"], taxonomy=taxonomy)
        assert "A real person" in result["entities"]["person"]


class TestSelectCandidateLabels:
    def test_returns_top_k_labels(self):
        from graphknows.ingestion.extraction.entities.label_space import select_candidate_labels

        label_space = {
            "PERSON": {"aliases": ["president"], "children": []},
            "ORGANIZATION": {"aliases": ["parliament"], "children": []},
            "LOCATION": {"aliases": ["ukraine"], "children": []},
            "POLICY": {"aliases": ["sanctions"], "children": []},
            "EVENT": {"aliases": ["invasion"], "children": []},
        }
        result = select_candidate_labels("The president signed a decree.", label_space, top_k=3)
        # Should return at most top_k
        assert len(result) <= 3
        # "president" is a hint for PERSON — should be in results
        assert "PERSON" in result

    def test_returns_top_k_even_with_no_matches(self):
        from graphknows.ingestion.extraction.entities.label_space import select_candidate_labels

        label_space = {
            "TYPE_A": {"aliases": [], "children": []},
            "TYPE_B": {"aliases": [], "children": []},
            "TYPE_C": {"aliases": [], "children": []},
        }
        result = select_candidate_labels("unrelated text xyz", label_space, top_k=2)
        assert len(result) == 2  # fallback to first 2

    def test_scores_alias_matches_higher(self):
        from graphknows.ingestion.extraction.entities.label_space import select_candidate_labels

        label_space = {
            "PERSON": {"aliases": ["prime minister"], "children": []},
            "UNRELATED": {"aliases": ["nothing"], "children": []},
        }
        result = select_candidate_labels("The prime minister spoke today.", label_space, top_k=1)
        assert result[0] == "PERSON"

    def test_child_token_scoring(self):
        from graphknows.ingestion.extraction.entities.label_space import select_candidate_labels

        label_space = {
            "LOCATION": {"aliases": [], "children": ["COUNTRY", "CITY"]},
            "POLICY": {"aliases": [], "children": []},
        }
        result = select_candidate_labels("The country declared independence.", label_space, top_k=1)
        assert result[0] == "LOCATION"


class TestRefineEntitiesWithOntology:
    def test_returns_entities_with_ontology_label(self):
        from graphknows.ingestion.extraction.entities.label_space import (
            refine_entities_with_ontology,
        )

        entities = [
            {"text": "Alice", "label": "PERSON", "confidence": 0.9},
        ]
        label_space = {"PERSON": {"children": [], "aliases": [], "child_aliases": {}}}
        result = refine_entities_with_ontology(entities, label_space=label_space, taxonomy=None)
        assert len(result) == 1
        assert "ontology_label" in result[0]

    def test_empty_entities_returns_empty(self):
        from graphknows.ingestion.extraction.entities.label_space import (
            refine_entities_with_ontology,
        )

        result = refine_entities_with_ontology([], label_space={}, taxonomy=None)
        assert result == []
