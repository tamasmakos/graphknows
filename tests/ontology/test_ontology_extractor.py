"""Tests for OntologyRelExtractor and Neo4j ONTOLOGY_REL write."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.ontology.loader import OntologyLoader
from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor
from graphknows.ingestion.ontology.router import OntologyRouter

REGISTRY_ROOT = Path(__file__).parent.parent.parent / "graphknows" / "ingestion" / "ontology" / "assets"


# ---------------------------------------------------------------------------
# OntologyRelExtractor tests — DSPy mocked, no real LLM calls
# ---------------------------------------------------------------------------

class TestOntologyRelExtractor:
    def _make_extractor_with_mock(self, relations: list[tuple]) -> OntologyRelExtractor:
        """Build an extractor whose RDF-backed delegate returns fixed relations."""
        extractor = OntologyRelExtractor.__new__(OntologyRelExtractor)
        extractor._threshold = 0.5
        extractor._extractor = MagicMock()
        extractor._extractor.extract_relations = AsyncMock(return_value=(relations, [], {}))
        return extractor

    @pytest.mark.asyncio
    async def test_extract_async_returns_rdf_validated_triples(self):
        extractor = self._make_extractor_with_mock(
            [
                ("EU", "REGULATES", "AI systems", {"confidence": 0.9})
            ]
        )
        result = await extractor.extract_async("The EU regulates AI systems.")

        assert result == [
            {
                "subject": "EU",
                "predicate": "REGULATES",
                "object": "AI systems",
                "confidence": 0.9,
            }
        ]
        extractor._extractor.extract_relations.assert_awaited_once_with(
            text="The EU regulates AI systems."
        )

    @pytest.mark.asyncio
    async def test_extract_async_filters_low_confidence_relations(self):
        extractor = self._make_extractor_with_mock(
            [
                ("A", "REL", "B", {"confidence": 0.4}),
                ("A", "REL", "C", {"confidence": 0.7}),
            ]
        )
        result = await extractor.extract_async("A relates to B and C")

        assert result == [{"subject": "A", "predicate": "REL", "object": "C", "confidence": 0.7}]

    def test_constructor_requires_ontology_file(self):
        with pytest.raises(RuntimeError, match="ontology_file"):
            OntologyRelExtractor("")

    def test_constructor_rejects_injected_lm(self):
        with pytest.raises(RuntimeError, match="injected LM"):
            OntologyRelExtractor("ontology.ttl", lm=MagicMock())


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
