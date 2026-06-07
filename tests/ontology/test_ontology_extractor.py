"""Tests for OntologyRelExtractor and Neo4j ONTOLOGY_REL write."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.ontology.loader import OntologyLoader
from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor
from graphknows.ingestion.ontology.router import OntologyRouter

REGISTRY_ROOT = Path(__file__).parent.parent.parent / "graphknows" / "ingestion" / "ontology" / "assets"


# ---------------------------------------------------------------------------
# OntologyRelExtractor tests — DSPy mocked, no real LLM calls
# ---------------------------------------------------------------------------

class TestOntologyRelExtractor:
    def _make_extractor_with_mock(
        self,
        relations: list[dict],
        relation_schema: dict[str, str] | None = None,
        entity_labels: list[str] | None = None,
    ) -> OntologyRelExtractor:
        """Build an extractor whose DSPy Predict module returns fixed relations."""
        extractor = OntologyRelExtractor.__new__(OntologyRelExtractor)
        extractor._threshold = 0.5
        extractor._relation_schema = relation_schema or {
            "REGULATES": "regulates",
            "IMPLEMENTS": "implements",
            "REL": "rel",
        }
        extractor._entity_labels = entity_labels or ["PERSON", "ORGANIZATION", "CONCEPT"]
        extractor._allowed_relations = {k: k for k in extractor._relation_schema}
        extractor._allowed_node_types = list(extractor._entity_labels)
        extractor._lm = MagicMock()

        mock_pred = MagicMock()
        mock_pred.relations = relations

        mock_predict = MagicMock(return_value=mock_pred)
        extractor._extractor = mock_predict
        return extractor

    def test_extract_returns_validated_triples(self):
        extractor = self._make_extractor_with_mock(
            [
                {"head": "EU", "relation": "REGULATES", "tail": "AI systems", "confidence": 0.9},
                {
                    "head": "Member States",
                    "relation": "IMPLEMENTS",
                    "tail": "Directive",
                    "confidence": 0.75,
                },
            ]
        )
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("lkif")

        result = extractor.extract(
            text="The EU regulates AI systems. Member States must implement the Directive.",
            entities=[
                {"name": "EU", "label": "LegalPerson"},
                {"name": "AI systems", "label": "CONCEPT"},
                {"name": "Member States", "label": "ORGANIZATION"},
                {"name": "Directive", "label": "CONCEPT"},
            ],
            ontology_schema=ontology.summary_jsonld,
            ontology_id="lkif",
        )
        assert len(result) == 2
        assert result[0]["predicate"] == "REGULATES"
        assert result[0]["confidence"] == pytest.approx(0.9)

    def test_extract_passes_dynamic_taxonomy_and_entities_to_dspy(self):
        extractor = self._make_extractor_with_mock(
            [{"head": "A", "relation": "SUPPORTS", "tail": "B", "confidence": 0.8}],
            relation_schema={"SUPPORTS": "supports"},
            entity_labels=["PERSON", "POLICY"],
        )
        result = extractor.extract(
            "A supports B",
            [{"name": "A", "type": "PERSON", "score": 0.9}, {"name": "B", "type": "POLICY"}],
            ontology_id="political",
        )
        assert len(result) == 1
        assert result[0]["subject"] == "A"
        call_kwargs = extractor._extractor.call_args.kwargs
        assert call_kwargs["text"] == "A supports B"
        assert call_kwargs["allowed_node_types"] == ["PERSON", "POLICY"]
        assert call_kwargs["allowed_relation_types"] == ["SUPPORTS"]
        assert call_kwargs["entities"] == [
            {"name": "A", "type": "PERSON", "score": 0.9},
            {"name": "B", "type": "POLICY", "score": 1.0},
        ]

    def test_extract_raises_on_malformed_output(self):
        extractor = self._make_extractor_with_mock([])
        extractor._extractor.return_value.relations = object()
        with pytest.raises(RuntimeError, match="unsupported relations type"):
            extractor.extract(
                "X relates to Y",
                [{"name": "X", "label": "Y"}, {"name": "Y", "label": "Y"}],
                ontology_id="schema-org",
            )

    def test_extract_returns_empty_when_no_entities(self):
        extractor = self._make_extractor_with_mock([])
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("fibo-fnd")
        result = extractor.extract("text", [], ontology.summary_jsonld, "fibo-fnd")
        assert result == []

    def test_extract_filters_incomplete_triples(self):
        extractor = self._make_extractor_with_mock(
            [
                {"head": "A", "relation": "REL", "tail": "B"},
                {"head": "A", "relation": "REL"},
                {"relation": "REL", "tail": "B"},
                {"head": "X", "tail": "Y"},
            ]
        )
        loader = OntologyLoader(REGISTRY_ROOT)
        ontology = loader.load("lkif")
        result = extractor.extract(
            "text",
            [{"name": "A", "label": "Z"}, {"name": "B", "label": "Z"}],
            ontology.summary_jsonld,
            "lkif",
        )
        assert len(result) == 1
        assert result[0]["subject"] == "A"

    def test_extract_rejects_unknown_relation_and_endpoint(self):
        extractor = self._make_extractor_with_mock(
            [
                {"head": "A", "relation": "UNKNOWN", "tail": "B", "confidence": 0.9},
                {"head": "A", "relation": "REL", "tail": "Ghost", "confidence": 0.9},
                {"head": "A", "relation": "REL", "tail": "B", "confidence": 0.9},
            ]
        )
        result = extractor.extract(
            "A relates to B",
            [{"name": "A", "label": "Z"}, {"name": "B", "label": "Z"}],
            "{}",
            "test",
        )
        assert result == [
            {"subject": "A", "predicate": "REL", "object": "B", "confidence": 0.9}
        ]

    def test_missing_openrouter_key_raises_in_llm_assisted_mode(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OntologyRelExtractor(relation_schema={"REL": "rel"}, entity_labels=["ENTITY"])


# ---------------------------------------------------------------------------
# Neo4j write_ontology_relations — mock driver, assert Cypher params
# ---------------------------------------------------------------------------

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
