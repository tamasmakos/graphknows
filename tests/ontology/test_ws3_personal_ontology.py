"""WS-3 tests — personal ontology, extract-then-map, router selection.

All tests are offline: no live API calls, no embedding model.  The LLM is
mocked wherever DSPyExtractor would normally require one.

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
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
ASSETS_ROOT = REPO_ROOT / "graphknows" / "ontology" / "assets"
PERSONAL_TTL = ASSETS_ROOT / "personal" / "personal.ttl"
PERSONAL_SUMMARY = ASSETS_ROOT / "personal" / "personal-summary.jsonld"

_ONTOLOGY_PKG = REPO_ROOT / "graphknows" / "ontology"
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
        "graphknows.ontology.loader",
        _ONTOLOGY_PKG / "loader.py",
    )


def _router_mod():
    # Router imports loader; ensure loader is pre-loaded in sys.modules first.
    _loader_mod()
    return _load_module(
        "graphknows.ontology.router",
        _ONTOLOGY_PKG / "router.py",
    )


def _filter_mod():
    # Relation filter imports loader lazily (inside functions) so no pre-load needed.
    _loader_mod()
    return _load_module(
        "graphknows.ingestion.extraction.relations.filter",
        _RELATIONS_PKG / "filter.py",
    )


def _utils_mod():
    return _load_module(
        "graphknows.ingestion.normalization",
        REPO_ROOT / "graphknows" / "ingestion" / "normalization.py",
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
            "Person", "Event", "Activity", "Preference", "Place", "Organization",
            "TimePoint", "Emotion", "Goal", "Possession", "Relationship",
        }
        missing = expected - labels
        assert not missing, f"Missing classes in personal.ttl: {missing}"

    def test_rdf_parse_yields_expected_properties(self):
        import rdflib
        from rdflib import OWL, RDF, RDFS

        graph = rdflib.Graph()
        graph.parse(str(PERSONAL_TTL))
        prop_nodes = set(graph.subjects(RDF.type, OWL.ObjectProperty))
        labels = {
            str(graph.value(n, RDFS.label)) for n in prop_nodes if graph.value(n, RDFS.label)
        }
        expected = {
            "lives_in", "works_as", "friend_of", "family_of", "partner_of",
            "attended", "owns", "likes", "dislikes", "plans_to",
            "happened_on", "located_in", "member_of",
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
            assert graph.value(prop, RDFS.domain) is not None, \
                f"Property {label} missing rdfs:domain"
            assert graph.value(prop, RDFS.range) is not None, \
                f"Property {label} missing rdfs:range"


# ============================================================
# 2. OntologyLoader — personal ontology registered and loadable
# ============================================================


class TestPersonalOntologyLoader:
    """Uses the isolated OntologyLoader against the bundled asset root."""

    def setup_method(self):
        mod = _loader_mod()
        self.OntologyLoader = mod.OntologyLoader
        self.OntologyManifest = mod.OntologyManifest
        self.load_rdf_ontology_file = mod.load_rdf_ontology_file
        self.loader = self.OntologyLoader(ASSETS_ROOT)

    def test_personal_in_registry(self):
        assert "personal" in self.loader.list_ids()

    def test_personal_loads(self):
        m = self.loader.load("personal")
        assert isinstance(m, self.OntologyManifest)
        assert m.id == "personal"

    def test_personal_has_classes(self):
        m = self.loader.load("personal")
        assert len(m.class_labels) >= 11, f"Got: {m.class_labels}"

    def test_personal_has_properties(self):
        m = self.loader.load("personal")
        assert len(m.property_labels) >= 13, f"Got: {m.property_labels}"

    def test_personal_has_keywords(self):
        m = self.loader.load("personal")
        assert len(m.keywords) >= 5, f"Got: {m.keywords}"

    def test_personal_summary_jsonld_valid(self):
        m = self.loader.load("personal")
        data = json.loads(m.summary_jsonld)
        assert isinstance(data, dict)

    def test_load_rdf_ontology_file_works_on_personal(self):
        manifest = self.load_rdf_ontology_file(PERSONAL_TTL)
        assert manifest.class_labels
        assert manifest.property_labels
        assert "friend_of" in manifest.property_labels
        assert "Person" in manifest.class_labels


# ============================================================
# 3. OntologyRouter — selects personal for conversational text
# ============================================================


class TestOntologyRouterPersonal:
    """Keyword-only routing (alpha=1.0, no embedder) for conversational snippets."""

    def setup_method(self):
        loader_mod = _loader_mod()
        router_mod = _router_mod()
        loader = loader_mod.OntologyLoader(ASSETS_ROOT)
        self.router = router_mod.OntologyRouter(loader=loader)
        self.router._ensure_ready()
        self.router._embedder = None
        self.router._desc_embeddings = None

    def test_conversational_text_selects_personal(self):
        text = (
            "Alice told me she's friends with Bob. She really likes hiking and dislikes crowded "
            "places. Her family moved to Berlin last summer and she plans to visit her partner "
            "in Madrid next year."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "personal", f"Expected 'personal', got '{m.id}'"

    def test_personal_in_select_multi(self):
        text = (
            "She met her friend at the birthday party last weekend "
            "and they talked about their families."
        )
        results = self.router.select_multi(text, top_k=3, alpha=1.0)
        ids = [r.id for r in results]
        assert "personal" in ids, f"'personal' not in top-3: {ids}"

    def test_select_rdf_file_returns_path_for_personal(self):
        text = (
            "He likes playing guitar, dislikes loud music. "
            "His friend lives in London and his family is very close."
        )
        path = self.router.select_rdf_file(text, alpha=1.0)
        assert path is not None
        assert path.endswith(".ttl"), f"Expected .ttl path, got: {path}"
        assert Path(path).exists(), f"Returned path does not exist: {path}"

    def test_legal_text_still_selects_lkif(self):
        text = (
            "The plaintiff filed a lawsuit under the statute. "
            "Jurisdiction was contested and the contract was deemed invalid."
        )
        m = self.router.select(text, alpha=1.0)
        assert m.id == "lkif", f"Expected lkif, got {m.id}"


# ============================================================
# 4. Extract-then-map: OntologyPredicateMapper
# ============================================================

PERSONAL_PROPS = [
    "lives_in", "works_as", "friend_of", "family_of", "partner_of",
    "attended", "owns", "likes", "dislikes", "plans_to",
    "happened_on", "located_in", "member_of",
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


# ============================================================
# 5. RelationOntologyFilter backward-compat
# ============================================================


class TestRelationOntologyFilterPersonal:
    def setup_method(self):
        mod = _filter_mod()
        self.RelationOntologyFilter = mod.RelationOntologyFilter
        self.build_predicate_mapper = mod.build_predicate_mapper

    def test_personal_filter_allows_friend_of(self):
        f = self.RelationOntologyFilter(["friend_of", "lives_in", "member_of"])
        assert f.is_allowed("friend_of")
        assert f.is_allowed("FRIEND_OF")
        assert f.is_allowed("friend of")

    def test_personal_filter_drops_unknown(self):
        f = self.RelationOntologyFilter(["friend_of", "lives_in"])
        assert not f.is_allowed("filed_complaint")

    def test_build_predicate_mapper_personal(self):
        """build_predicate_mapper should work for the registered personal ontology."""
        mapper = self.build_predicate_mapper("personal")
        assert mapper.ontology_labels, "Personal predicate mapper should have labels"
        assert "friend_of" in mapper.ontology_labels


# ============================================================
# 6. DSPy extractor — extract-then-map (mocked LLM)
# ============================================================


class TestDspyExtractorExtractThenMap:
    """Verify the extract-then-map step in DSPyExtractor without live API."""

    def _make_extractor(self):
        """Build a DSPyExtractor with personal ontology, mocking the DSPy LM."""
        # Pre-load dependencies that dspy.py needs so they don't go through extraction __init__
        _loader_mod()
        _filter_mod()

        # Load utils module the extractor depends on
        _load_module(
            "graphknows.ingestion.normalization",
            REPO_ROOT / "graphknows" / "ingestion" / "normalization.py",
        )
        _load_module(
            "graphknows.ingestion.extraction.entities.graph_ops",
            REPO_ROOT / "graphknows" / "ingestion" / "extraction" / "entities" / "graph_ops.py",
        )

        with (
            patch("dspy.LM", return_value=MagicMock()),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        ):
            dspy_mod = _load_module(
                "graphknows.ingestion.extraction.relations.llm_assisted.dspy",
                _RELATIONS_PKG / "llm_assisted" / "dspy.py",
            )
            config = {
                "llm": {
                    "extraction_model": "deepseek/deepseek-v4-flash",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
                "ontology": {"file": str(PERSONAL_TTL)},
            }
            return dspy_mod.DSPyExtractor(config)

    def test_extractor_loads_personal_ontology(self):
        extractor = self._make_extractor()
        all_types = [t.lower() for t in extractor._ontology_relation_types]
        assert "friend_of" in all_types or "FRIEND_OF" in extractor._ontology_relation_types, \
            f"friend_of not in relation types: {extractor._ontology_relation_types}"

    def test_extractor_has_predicate_mapper(self):
        filter_mod = _filter_mod()
        extractor = self._make_extractor()
        assert isinstance(extractor._predicate_mapper, filter_mod.OntologyPredicateMapper)
        assert extractor._predicate_mapper.ontology_labels

    def test_apply_extract_then_map_remaps_surface_form(self):
        extractor = self._make_extractor()
        raw = {
            "source": "Alice",
            "relation": "is friends with",
            "target": "Bob",
            "source_type": "Person",
            "target_type": "Person",
            "evidence": "Alice is friends with Bob.",
        }
        mapped_raw, drop_reason = extractor._apply_extract_then_map(raw)
        assert mapped_raw is not None, f"Expected mapping, got drop_reason={drop_reason}"
        assert mapped_raw["relation"] == "friend_of", f"Got: {mapped_raw['relation']}"
        assert "_mapped_from" in mapped_raw
        assert mapped_raw["_mapped_from"] == "is friends with"

    def test_apply_extract_then_map_rejects_off_ontology(self):
        extractor = self._make_extractor()
        raw = {
            "source": "Corp",
            "relation": "filed_derivative_contract",
            "target": "Fund",
            "source_type": "Organization",
            "target_type": "Organization",
            "evidence": "Corp filed a derivative contract.",
        }
        mapped_raw, drop_reason = extractor._apply_extract_then_map(raw)
        assert mapped_raw is None, \
            f"Expected rejection but got mapped_raw={mapped_raw}"
        assert drop_reason is not None

    def test_apply_extract_then_map_exact_match_passthrough(self):
        extractor = self._make_extractor()
        raw = {
            "source": "Alice",
            "relation": "lives_in",
            "target": "Berlin",
            "source_type": "Person",
            "target_type": "Place",
            "evidence": "Alice lives in Berlin.",
        }
        mapped_raw, drop_reason = extractor._apply_extract_then_map(raw)
        assert mapped_raw is not None
        assert mapped_raw["relation"] == "lives_in"
        # No remapping for exact match — no _mapped_from key
        assert "_mapped_from" not in mapped_raw

    def test_enriched_ontology_context_contains_properties_detail(self):
        extractor = self._make_extractor()
        ctx = extractor._ontology_enriched_context
        assert ctx, "Enriched context should not be empty"
        data = json.loads(ctx)
        assert "properties_detail" in data, \
            f"Missing 'properties_detail' key in context: {list(data.keys())}"
        # Each property detail should have label, domain, range
        for entry in data["properties_detail"]:
            assert "label" in entry
            assert "domain" in entry, f"Property {entry.get('label')} missing domain"
            assert "range" in entry, f"Property {entry.get('label')} missing range"
