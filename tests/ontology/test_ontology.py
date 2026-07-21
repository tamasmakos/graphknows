"""Tests for the OntologyLoader and OntologyRouter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphknows.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ontology.router import OntologyRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGISTRY_ROOT = (
    Path(__file__).parent.parent.parent / "graphknows" / "ontology" / "assets"
)


# ---------------------------------------------------------------------------
# OntologyLoader tests
# ---------------------------------------------------------------------------


class TestOntologyLoader:
    def setup_method(self):
        self.loader = OntologyLoader(REGISTRY_ROOT)

    def test_registry_loads(self):
        ids = self.loader.list_ids()
        assert len(ids) >= 5, f"Expected >= 5 ontologies, got {ids}"

    def test_all_expected_ids_present(self):
        ids = self.loader.list_ids()
        for expected in ["lkif", "prov-o", "schema-org", "fibo-fnd", "political"]:
            assert expected in ids, f"'{expected}' missing from registry"

    def test_load_lkif_returns_manifest(self):
        m = self.loader.load("lkif")
        assert isinstance(m, OntologyManifest)
        assert m.id == "lkif"
        assert len(m.class_labels) >= 10, f"LKIF should have >=10 classes, got {m.class_labels}"

    def test_load_prov_o_returns_manifest(self):
        m = self.loader.load("prov-o")
        assert isinstance(m, OntologyManifest)
        assert len(m.class_labels) >= 5

    def test_load_schema_org_returns_manifest(self):
        m = self.loader.load("schema-org")
        assert isinstance(m, OntologyManifest)
        assert len(m.class_labels) >= 10

    def test_load_fibo_returns_manifest(self):
        m = self.loader.load("fibo-fnd")
        assert isinstance(m, OntologyManifest)
        assert len(m.class_labels) >= 10

    def test_load_political_returns_manifest(self):
        m = self.loader.load("political")
        assert isinstance(m, OntologyManifest)
        assert len(m.class_labels) >= 10

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.loader.load("nonexistent_ontology_xyz")

    def test_caching(self):
        m1 = self.loader.load("lkif")
        m2 = self.loader.load("lkif")
        assert m1 is m2  # Same object from cache

    def test_summary_jsonld_is_valid_json(self):
        for oid in self.loader.list_ids():
            m = self.loader.load(oid)
            data = json.loads(m.summary_jsonld)
            assert isinstance(data, dict), f"{oid} summary_jsonld is not a JSON object"

    def test_manifest_has_keywords(self):
        for oid in self.loader.list_ids():
            m = self.loader.load(oid)
            assert len(m.keywords) >= 5, f"{oid} should have >=5 keywords"


# ---------------------------------------------------------------------------
# OntologyRouter tests (keyword-only — no embedding model dependency)
# ---------------------------------------------------------------------------


class TestOntologyRouterKeyword:
    """Test routing using keyword heuristic only (alpha=1.0)."""

    def setup_method(self):
        loader = OntologyLoader(REGISTRY_ROOT)
        self.router = OntologyRouter(loader=loader)
        # Force keyword-only by patching embed model to None
        self.router._ensure_ready()
        self.router._embed_model = None
        self.router._desc_embeddings = None

    def test_legal_text_selects_lkif(self):
        text = (
            "The European Parliament adopted a directive regulating AI liability. "
            "The plaintiff argued that the statute violated constitutional provisions. "
            "Jurisdiction was contested in this contract dispute."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "lkif", f"Expected lkif, got {m.id}"

    def test_finance_text_selects_fibo(self):
        text = (
            "Investors saw significant returns from the stock portfolio. "
            "Bond yields rose sharply amid market volatility. "
            "The bank announced a new derivative financial instrument."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "fibo-fnd", f"Expected fibo-fnd, got {m.id}"

    def test_political_text_selects_political(self):
        text = (
            "The parliament voted on the coalition agreement. "
            "The prime minister appointed a new minister of foreign affairs. "
            "The opposition party boycotted the election results."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "political", f"Expected political, got {m.id}"

    def test_select_multi_returns_top_k(self):
        text = "The legislation passed by parliament regulates financial instruments."
        results = self.router.select_multi(text, top_k=2, alpha=1.0)
        assert len(results) == 2
        ids = [r.id for r in results]
        # Should contain at least one of lkif or political (legislation/parliament keywords)
        assert any(i in ids for i in ["lkif", "political"]), f"Got: {ids}"

    def test_select_multi_unique(self):
        text = "stock contract market law"
        results = self.router.select_multi(text, top_k=3, alpha=1.0)
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids)), "select_multi returned duplicates"


# ---------------------------------------------------------------------------
# Router integration: assert right ontology selected per domain text
# ---------------------------------------------------------------------------


class TestOntologyRouterIntegration:
    def setup_method(self):
        loader = OntologyLoader(REGISTRY_ROOT)
        self.router = OntologyRouter(loader=loader)
        self.router._ensure_ready()
        self.router._embed_model = None
        self.router._desc_embeddings = None

    def test_science_text_selects_schema_org(self):
        text = "The research paper was published by the institution. The dataset was authored by multiple researchers."
        m = self.router.select(text, alpha=1.0)
        assert m.id == "schema-org", f"Got {m.id}"

    def test_provenance_text_selects_prov_o(self):
        text = (
            "The activity was attributed to the agent. The organization delegated responsibility."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "prov-o", f"Got {m.id}"


# ---------------------------------------------------------------------------
# Extended ontology tests
# ---------------------------------------------------------------------------


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
