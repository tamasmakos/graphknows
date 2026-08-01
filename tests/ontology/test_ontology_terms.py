"""Ontology terms + the definition-embedding catalog.

The definitions are the point: the label lists already existed, but nothing
parsed the ``skos:definition`` text that makes a class reachable from prose
which never says its label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphknows.ontology.catalog import OntologyIndex, load_ontology_index, reset_ontology_indexes
from graphknows.ontology.loader import OntologyTerm, load_ontology_terms

ASSETS = Path(__file__).resolve().parents[2] / "graphknows" / "ontology" / "assets"
PERSONAL_TTL = ASSETS / "personal" / "personal.ttl"


class TestLoadOntologyTerms:
    def test_single_file_yields_classes_and_properties(self) -> None:
        terms = load_ontology_terms(PERSONAL_TTL)
        classes = [t for t in terms if t.kind == "class"]
        properties = [t for t in terms if t.kind == "property"]
        assert len(classes) == 11
        assert len(properties) == 13
        assert {t.label for t in classes} >= {"Person", "Event", "Place", "Activity"}

    def test_every_personal_term_has_a_definition(self) -> None:
        """personal.ttl is fully annotated; a regression here means SKOS parsing broke."""
        terms = load_ontology_terms(PERSONAL_TTL)
        assert all(t.definition for t in terms)
        person = next(t for t in terms if t.label == "Person")
        assert "human individual" in person.definition

    def test_domain_and_range_are_parsed(self) -> None:
        terms = load_ontology_terms(PERSONAL_TTL)
        attended = next(t for t in terms if t.label == "attended")
        assert attended.kind == "property"
        # Multi-valued even for a single rdfs:domain — see OntologyTerm.
        assert attended.domain == ("Person",)
        assert attended.range == ("Event",)

    def test_folder_loads_every_ontology_beneath_it(self) -> None:
        one = load_ontology_terms(PERSONAL_TTL)
        several = load_ontology_terms(ASSETS)
        assert len(several) > len(one) * 5  # the bundle spans 6 ontologies
        assert {t.source for t in several} != {str(PERSONAL_TTL)}

    def test_folder_dedupes_by_label_not_uri(self) -> None:
        """personal/ holds the TTL and its JSON-LD digest; labels must not double."""
        terms = load_ontology_terms(ASSETS / "personal")
        keys = [(t.kind, t.label.casefold()) for t in terms]
        assert len(keys) == len(set(keys))

    def test_missing_source_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ontology_terms(ASSETS / "does-not-exist")

    def test_source_without_terms_raises(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not an ontology", encoding="utf-8")
        with pytest.raises(ValueError, match="No ontology classes"):
            load_ontology_terms(tmp_path)

    def test_index_text_combines_label_and_definition(self) -> None:
        term = OntologyTerm(uri="u", label="Place", definition="A named location.", kind="class")
        assert term.index_text == "Place. A named location"


class TestOntologyIndex:
    @pytest.fixture(autouse=True)
    def _clean(self) -> None:
        reset_ontology_indexes()

    def test_empty_source_disables_injection_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Injection is always on by default, so "off" must announce itself."""
        with caplog.at_level("WARNING", logger="graphknows.ontology.catalog"):
            assert load_ontology_index("") is None
        assert "Ontology injection is OFF" in caplog.text

    def test_unusable_source_is_not_fatal(self, tmp_path: Path) -> None:
        """Ontology injection is enrichment: a bad path must not fail ingestion."""
        assert load_ontology_index(str(tmp_path / "nope.ttl")) is None

    def test_index_is_cached_per_source(self) -> None:
        a = load_ontology_index(str(PERSONAL_TTL))
        b = load_ontology_index(str(PERSONAL_TTL))
        assert a is b

    def test_classes_and_properties_are_separate_matrices(self) -> None:
        idx = load_ontology_index(str(PERSONAL_TTL))
        assert isinstance(idx, OntologyIndex)
        assert len(idx.classes) == 11
        assert len(idx.properties) == 13
        assert set(idx.class_labels).isdisjoint(idx.property_labels)

    def test_match_classes_never_returns_a_property(self) -> None:
        idx = load_ontology_index(str(PERSONAL_TTL))
        assert idx is not None
        emb = np.random.default_rng(0).normal(size=384).tolist()
        assert all(t.kind == "class" for t, _ in idx.match_classes(emb, top_k=5))
        assert all(t.kind == "property" for t, _ in idx.match_properties(emb, top_k=5))

    def test_match_is_ranked_and_top_k_bounded(self) -> None:
        idx = load_ontology_index(str(PERSONAL_TTL))
        assert idx is not None
        from graphknows.storage.embedder import embed_one

        hits = idx.match_classes(embed_one("We met at the concert last night."), top_k=3)
        assert len(hits) <= 3
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    def test_degenerate_embedding_matches_nothing(self) -> None:
        idx = load_ontology_index(str(PERSONAL_TTL))
        assert idx is not None
        assert idx.match_classes([]) == []
        assert idx.match_classes([0.0] * 384) == []
        assert idx.match_classes([0.1, 0.2]) == []  # wrong dimensionality

    def test_every_term_self_matches(self) -> None:
        """A term's own index_text must retrieve that term — not its neighbour.

        Regression: the matrix row order comes from the term list, but terms were
        built from an rdflib subject SET (process-randomised order) and the disk
        cache validated only the row COUNT. A cached matrix then paired with a
        differently-ordered term list, returning cosine 1.0 for the wrong label.
        """
        from graphknows.storage.embedder import embed_one

        idx = load_ontology_index(str(PERSONAL_TTL))
        assert idx is not None
        for term in idx.classes:
            hit = idx.match_classes(embed_one(term.index_text), top_k=1, min_sim=0.0)
            assert hit and hit[0][0].label == term.label, (
                f"{term.label} self-matched to {hit[0][0].label if hit else None}"
            )
        for term in idx.properties:
            hit = idx.match_properties(embed_one(term.index_text), top_k=1, min_sim=0.0)
            assert hit and hit[0][0].label == term.label

    def test_term_order_is_deterministic_across_loads(self) -> None:
        """Row order must not depend on set iteration, or the cache misaligns."""
        first = [t.uri for t in load_ontology_terms(PERSONAL_TTL)]
        second = [t.uri for t in load_ontology_terms(PERSONAL_TTL)]
        assert first == second
        # Classes precede properties, each block internally sorted by URI.
        terms = load_ontology_terms(PERSONAL_TTL)
        class_uris = [t.uri for t in terms if t.kind == "class"]
        assert class_uris == sorted(class_uris)

    def test_term_embedding_is_a_copy_not_a_recompute(self) -> None:
        idx = load_ontology_index(str(PERSONAL_TTL))
        assert idx is not None
        vec = idx.term_embedding(idx.classes[0].uri)
        assert len(vec) == 384
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4  # matrix rows are normalized
        assert idx.term_embedding("urn:not-a-real-uri") == []
