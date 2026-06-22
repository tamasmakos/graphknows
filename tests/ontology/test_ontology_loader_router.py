"""Tests for the OntologyLoader and OntologyRouter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphknows.extraction.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.extraction.ontology.router import OntologyRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGISTRY_ROOT = (
    Path(__file__).parent.parent.parent / "graphknows" / "extraction" / "ontology" / "assets"
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
