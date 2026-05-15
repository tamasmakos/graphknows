"""Tests for OntologyRelExtractor and Neo4j ONTOLOGY_REL write."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "agent"))

from ontology.loader import OntologyLoader
from ontology.rel_extractor import OntologyRelExtractor
from ontology.router import OntologyRouter

REGISTRY_ROOT = Path(__file__).parent.parent / "services" / "agent" / "ontology" / "assets"


# ---------------------------------------------------------------------------
# OntologyRelExtractor tests — DSPy mocked, no real LLM calls
# ---------------------------------------------------------------------------

class TestOntologyRelExtractor:
    def _make_extractor_with_mock(self, relations_json: str) -> OntologyRelExtractor:
        """Build an extractor whose DSPy ChainOfThought returns a fixed prediction."""
        extractor = OntologyRelExtractor.__new__(OntologyRelExtractor)
        extractor._available = True
        extractor._dspy = MagicMock()

        mock_pred = MagicMock()
        mock_pred.relations_json = relations_json

        mock_cot = MagicMock()
        mock_cot.return_value = mock_pred
        extractor._extractor = mock_cot
        return extractor

    def test_extract_returns_validated_triples(self):
        payload = json.dumps([
            {"subject": "EU", "predicate": "regulates", "object": "AI systems", "confidence": 0.9},
            {"subject": "Member States", "predicate": "implements", "object": "Directive", "confidence": 0.75},
        ])
        extractor = self._make_extractor_with_mock(payload)
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("lkif")

        result = extractor.extract(
            text="The EU regulates AI systems. Member States must implement the Directive.",
            entities=[{"name": "EU", "label": "LegalPerson"}, {"name": "AI systems", "label": "CONCEPT"}],
            ontology_schema=ontology.summary_jsonld,
            ontology_id="lkif",
        )
        assert len(result) == 2
        assert result[0]["predicate"] == "regulates"
        assert result[0]["confidence"] == pytest.approx(0.9)

    def test_extract_strips_markdown_fences(self):
        payload = "```json\n[{\"subject\": \"A\", \"predicate\": \"rel\", \"object\": \"B\", \"confidence\": 0.5}]\n```"
        extractor = self._make_extractor_with_mock(payload)
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("political")
        result = extractor.extract("A relates to B", [{"name": "A", "label": "X"}], ontology.summary_jsonld, "political")
        assert len(result) == 1
        assert result[0]["subject"] == "A"

    def test_extract_returns_empty_on_bad_json(self):
        extractor = self._make_extractor_with_mock("not valid json at all")
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("schema-org")
        result = extractor.extract("text", [{"name": "X", "label": "Y"}], ontology.summary_jsonld, "schema-org")
        assert result == []

    def test_extract_returns_empty_when_no_entities(self):
        extractor = self._make_extractor_with_mock("[]")
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("fibo-fnd")
        result = extractor.extract("text", [], ontology.summary_jsonld, "fibo-fnd")
        assert result == []

    def test_extract_filters_incomplete_triples(self):
        payload = json.dumps([
            {"subject": "A", "predicate": "rel", "object": "B"},        # valid
            {"subject": "A", "predicate": "rel"},                        # missing object
            {"predicate": "rel", "object": "B"},                         # missing subject
            {"subject": "X", "object": "Y"},                             # missing predicate
        ])
        extractor = self._make_extractor_with_mock(payload)
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("lkif")
        result = extractor.extract("text", [{"name": "A", "label": "Z"}], ontology.summary_jsonld, "lkif")
        assert len(result) == 1
        assert result[0]["subject"] == "A"

    def test_extract_unavailable_returns_empty(self):
        extractor = OntologyRelExtractor.__new__(OntologyRelExtractor)
        extractor._available = False
        result = extractor.extract("text", [{"name": "X", "label": "Y"}], "{}", "test")
        assert result == []


# ---------------------------------------------------------------------------
# Neo4j write_ontology_relations — mock driver, assert Cypher params
# ---------------------------------------------------------------------------

class TestNeo4jOntologyWrite:
    def test_write_ontology_relations_calls_driver(self):
        """Assert write_ontology_relations executes the expected Cypher."""
        import asyncio
        from graph.backends.neo4j_gds import Neo4jGDSBackend

        backend = Neo4jGDSBackend.__new__(Neo4jGDSBackend)

        # Capture all run() calls
        run_calls: list[tuple] = []

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run = MagicMock(side_effect=lambda q, **kw: run_calls.append((q, kw)) or MagicMock())

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        backend._driver = mock_driver

        triples = [
            {"subject": "EU", "predicate": "regulates", "object": "AI", "confidence": 0.9},
        ]
        asyncio.run(backend.write_ontology_relations("sess-1", triples, "lkif"))

        assert len(run_calls) == 1
        _q, kw = run_calls[0]
        assert kw["subj"] == "EU"
        assert kw["obj"] == "AI"
        assert kw["pred"] == "regulates"
        assert kw["ont"] == "lkif"
        assert kw["sid"] == "sess-1"
        assert kw["conf"] == pytest.approx(0.9)

    def test_write_ontology_relations_skips_empty(self):
        import asyncio
        from graph.backends.neo4j_gds import Neo4jGDSBackend

        backend = Neo4jGDSBackend.__new__(Neo4jGDSBackend)
        mock_driver = MagicMock()
        backend._driver = mock_driver

        # Should return without calling driver.session at all
        asyncio.run(backend.write_ontology_relations("sess-1", [], "lkif"))
        mock_driver.session.assert_not_called()


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
        text = "The activity was attributed to the agent. The organization delegated responsibility."
        m = self.router.select(text, alpha=1.0)
        assert m.id == "prov-o", f"Got {m.id}"
