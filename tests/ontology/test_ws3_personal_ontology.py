"""WS-3 tests — personal ontology and extract-then-map.

All tests are offline: no live API calls, no embedding model.

Import strategy: to avoid the heavy ``graphknows.extraction.__init__`` (which
pulls in gliner/networkx/langchain etc.), sub-modules are loaded directly via
``importlib`` or absolute file paths.  The ``sys.modules`` cache ensures each
module is loaded at most once per session.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
ASSETS_ROOT = REPO_ROOT / "tests" / "fixtures" / "ontology"
PERSONAL_TTL = ASSETS_ROOT / "personal.ttl"
PERSONAL_SUMMARY = ASSETS_ROOT / "personal-summary.jsonld"

_ONTOLOGY_PKG = REPO_ROOT / "graphknows" / "symbolic" / "ontology"
_RELATIONS_PKG = REPO_ROOT / "graphknows" / "ingestion" / "extraction" / "relations"


# ---------------------------------------------------------------------------
# Isolated module loader — avoids heavy extraction __init__ imports
# ---------------------------------------------------------------------------


def _load_module(name: str, path: Path) -> Any:
    """Load a Python module from *path* with the given dotted *name*.

    Results are cached in sys.modules so repeated calls return the same object.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _loader_mod():
    return _load_module(
        "graphknows.symbolic.ontology.loader",
        _ONTOLOGY_PKG / "loader.py",
    )


def _filter_mod():
    # Relation filter imports loader lazily (inside functions) so no pre-load needed.
    _loader_mod()
    return _load_module(
        "graphknows.ingestion.extraction.relations.filter",
        _RELATIONS_PKG / "filter.py",
    )


# ============================================================
# 1. Personal ontology — file existence and RDF parse
# ============================================================


class TestPersonalOntologyFiles:
    def test_ttl_file_exists(self):
        assert PERSONAL_TTL.exists(), f"personal.ttl not found at {PERSONAL_TTL}"

    def test_summary_jsonld_exists(self):
        assert PERSONAL_SUMMARY.exists(), f"personal-summary.jsonld not found at {PERSONAL_SUMMARY}"

    def test_summary_jsonld_is_valid_json(self):
        data = json.loads(PERSONAL_SUMMARY.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_summary_has_classes(self):
        data = json.loads(PERSONAL_SUMMARY.read_text(encoding="utf-8"))
        assert len(data.get("classes", [])) >= 11

    def test_summary_has_properties(self):
        data = json.loads(PERSONAL_SUMMARY.read_text(encoding="utf-8"))
        assert len(data.get("properties", [])) >= 13

    def test_rdf_parse_yields_expected_classes(self):
        import rdflib
        from rdflib import OWL, RDF, RDFS

        graph = rdflib.Graph()
        graph.parse(str(PERSONAL_TTL))
        class_nodes = set(graph.subjects(RDF.type, OWL.Class)) | set(
            graph.subjects(RDF.type, RDFS.Class)
        )
        labels = {
            str(graph.value(n, RDFS.label)) for n in class_nodes if graph.value(n, RDFS.label)
        }
        expected = {
            "Person",
            "Event",
            "Activity",
            "Preference",
            "Place",
            "Organization",
            "TimePoint",
            "Emotion",
            "Goal",
            "Possession",
            "Relationship",
        }
        missing = expected - labels
        assert not missing, f"Missing classes in personal.ttl: {missing}"

    def test_rdf_parse_yields_expected_properties(self):
        import rdflib
        from rdflib import OWL, RDF, RDFS

        graph = rdflib.Graph()
        graph.parse(str(PERSONAL_TTL))
        prop_nodes = set(graph.subjects(RDF.type, OWL.ObjectProperty))
        labels = {str(graph.value(n, RDFS.label)) for n in prop_nodes if graph.value(n, RDFS.label)}
        expected = {
            "lives_in",
            "works_as",
            "friend_of",
            "family_of",
            "partner_of",
            "attended",
            "owns",
            "likes",
            "dislikes",
            "plans_to",
            "happened_on",
            "located_in",
            "member_of",
        }
        missing = expected - labels
        assert not missing, f"Missing properties in personal.ttl: {missing}"

    def test_properties_have_domain_and_range(self):
        import rdflib
        from rdflib import OWL, RDF, RDFS

        graph = rdflib.Graph()
        graph.parse(str(PERSONAL_TTL))
        prop_nodes = set(graph.subjects(RDF.type, OWL.ObjectProperty))
        for prop in prop_nodes:
            label = graph.value(prop, RDFS.label)
            assert graph.value(prop, RDFS.domain) is not None, (
                f"Property {label} missing rdfs:domain"
            )
            assert graph.value(prop, RDFS.range) is not None, f"Property {label} missing rdfs:range"


# ============================================================
# 2. Extract-then-map: OntologyPredicateMapper
# ============================================================

PERSONAL_PROPS = [
    "lives_in",
    "works_as",
    "friend_of",
    "family_of",
    "partner_of",
    "attended",
    "owns",
    "likes",
    "dislikes",
    "plans_to",
    "happened_on",
    "located_in",
    "member_of",
]


class TestOntologyPredicateMapper:
    """Offline tests for the character-n-gram similarity predicate mapper."""

    def setup_method(self):
        mod = _filter_mod()
        self.OntologyPredicateMapper = mod.OntologyPredicateMapper
        self.mapper = self.OntologyPredicateMapper(PERSONAL_PROPS)

    # --- exact-match cases ---

    def test_exact_match_friend_of(self):
        canonical, score = self.mapper.map_predicate("friend_of")
        assert canonical == "friend_of"
        assert score == 1.0

    def test_exact_match_case_insensitive(self):
        canonical, score = self.mapper.map_predicate("FRIEND_OF")
        assert canonical == "friend_of"
        assert score == 1.0

    def test_exact_match_punctuation_insensitive(self):
        canonical, score = self.mapper.map_predicate("friend of")
        assert canonical == "friend_of"
        assert score == 1.0

    # --- similarity-mapped cases ---

    def test_free_form_maps_friend(self):
        """'is friends with' should map to friend_of."""
        canonical, score = self.mapper.map_predicate("is friends with")
        assert canonical == "friend_of", f"Got '{canonical}' (score={score:.3f})"
        assert score > 0.0

    def test_free_form_maps_lives_in(self):
        canonical, score = self.mapper.map_predicate("resides in")
        assert canonical == "lives_in", f"Got '{canonical}' (score={score:.3f})"

    def test_free_form_maps_member_of(self):
        canonical, score = self.mapper.map_predicate("belongs to")
        assert canonical == "member_of", f"Got '{canonical}' (score={score:.3f})"

    # --- rejection cases ---

    def test_off_ontology_predicate_rejected(self):
        """A purely financial predicate should be unmappable from the personal ontology."""
        canonical, score = self.mapper.map_predicate("filed_derivative_contract")
        # Should be None (below threshold)
        if canonical is not None:
            pytest.fail(
                f"Expected 'filed_derivative_contract' to be unmappable from personal ontology, "
                f"but got '{canonical}' with score={score:.3f}"
            )

    def test_empty_labels_returns_none(self):
        mapper = self.OntologyPredicateMapper([])
        canonical, score = mapper.map_predicate("friend_of")
        assert canonical is None
        assert score == 0.0

    # --- map_relations batch ---

    def test_map_relations_remaps_and_drops(self):
        relations = [
            {"relation": "is friends with", "source": "Alice", "target": "Bob"},
            {"relation": "filed_derivative_contract", "source": "X", "target": "Y"},
            {"relation": "lives_in", "source": "Alice", "target": "Berlin"},
        ]
        result = self.mapper.map_relations(relations, label_key="relation")
        kept_relations = [r["relation"] for r in result]
        assert "friend_of" in kept_relations, f"friend_of not in {kept_relations}"
        assert "lives_in" in kept_relations
        # filed_derivative_contract should be dropped
        sources = [r["source"] for r in result]
        assert "X" not in sources, "Off-ontology relation should have been dropped"

    def test_map_relations_adds_mapped_from(self):
        relations = [{"relation": "is friends with", "source": "A", "target": "B"}]
        result = self.mapper.map_relations(relations)
        assert result, "Expected at least one mapped relation"
        assert result[0]["relation"] == "friend_of"
        assert "_mapped_from" in result[0]
        assert result[0]["_mapped_from"] == "is friends with"
