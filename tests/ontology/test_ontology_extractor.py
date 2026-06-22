"""Tests for ontology routing integration."""

from __future__ import annotations

from pathlib import Path

from graphknows.extraction.ontology.loader import OntologyLoader
from graphknows.extraction.ontology.router import OntologyRouter

REGISTRY_ROOT = (
    Path(__file__).parent.parent.parent / "graphknows" / "extraction" / "ontology" / "assets"
)


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
