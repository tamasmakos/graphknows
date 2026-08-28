"""Lexicalization must not hand BFO plumbing everyday triggers.

CCO ships no ``skos:altLabel``, so ``lexicalize`` harvests them from the label's
content words plus their dominant WordNet synset. For an UPPER-ONTOLOGY property
that synset is a common English verb, and the property ends up firing on
ordinary conversation while carrying none of its declared meaning.

Measured on conv-30: `permits` (Process Regulation -> process) acquired
"allow"/"let" and produced 23 edges — `Gina permits fashion`, `dancing permits
Gina`. With its siblings that was 64 of 286 REL edges (22%), none of them
asserted by anyone.

The prefLabel and the label's own words stay: text that says "permits" IS
asserting the property.
"""

from __future__ import annotations

import pytest

from graphknows.symbolic.ontology import skos
from graphknows.symbolic.ontology.skos import (
    Concept,
    _upper_ontology_property,
    _wordnet_synonyms,
    ancestors,
    lexicalize,
)


def _prop(label: str, domain: tuple[str, ...], rng: tuple[str, ...]) -> Concept:
    return Concept(
        uri=f"cco:{label}",
        pref_label=label,
        definition="",
        kind="property",
        domain=domain,
        range=rng,
    )


class TestUpperOntologyDetection:
    @pytest.mark.parametrize(
        ("label", "domain", "rng"),
        [
            ("permits", ("Process Regulation",), ("process",)),
            ("requires", ("Process Regulation",), ("process",)),
            ("is required by", ("process",), ("Process Regulation",)),
            ("realizes", ("process",), ("realizable entity",)),
            ("exists at", ("entity",), ("temporal region",)),
        ],
    )
    def test_bfo_typed_properties_are_upper_ontology(
        self, label: str, domain: tuple[str, ...], rng: tuple[str, ...]
    ) -> None:
        assert _upper_ontology_property(_prop(label, domain, rng))

    @pytest.mark.parametrize(
        ("label", "domain", "rng"),
        [
            ("works for", ("Person",), ("Organization",)),
            ("teaches", ("Person",), ("Person",)),
            ("owns", ("Person",), ("Material Artifact",)),
            ("lives in", ("Person",), ("Facility",)),
        ],
    )
    def test_concrete_properties_are_not(
        self, label: str, domain: tuple[str, ...], rng: tuple[str, ...]
    ) -> None:
        assert not _upper_ontology_property(_prop(label, domain, rng))

    def test_classes_are_never_upper_ontology_properties(self) -> None:
        """The guard is about PROPERTIES; a class keeps its synonyms regardless."""
        cls = Concept(uri="cco:X", pref_label="Grocery Store", definition="", kind="class")
        assert not _upper_ontology_property(cls)


class TestLexicalizeDropsPlumbingSynonyms:
    def test_permits_keeps_its_label_and_loses_allow_and_let(self) -> None:
        """The exact conv-30 defect: `Gina permits fashion`, minted from "let"."""
        c = lexicalize(_prop("permits", ("Process Regulation",), ("process",)))
        assert "permits" in c.alt_labels
        assert not {"allow", "let", "countenance", "permit"} & c.alt_labels

    def test_realizes_loses_recognize(self) -> None:
        c = lexicalize(_prop("realizes", ("process",), ("realizable entity",)))
        assert "realizes" in c.alt_labels
        assert not {"recognize", "recognise", "realise", "agnize"} & c.alt_labels

    def test_a_concrete_property_keeps_its_synonyms(self) -> None:
        """The guard must not cost the authored vocabulary its lexical surface."""
        c = lexicalize(_prop("teaches", ("Person",), ("Person",)))
        assert "teaches" in c.alt_labels
        # WordNet's dominant synset for the verb "teach" supplies these; if the
        # guard over-fires the conversational vocabulary goes silent.
        assert len(c.alt_labels) > 1

    def test_synonyms_disabled_is_still_honoured(self) -> None:
        c = lexicalize(_prop("teaches", ("Person",), ("Person",)), synonyms=False)
        assert c.alt_labels == {"teaches"}


def _cls(label: str, broader: tuple[str, ...] = ()) -> Concept:
    return Concept(
        uri=f"cco:{label}", pref_label=label, definition="", kind="class", broader=broader
    )


class TestMissingWordnetCorpus:
    """Lexicalization must degrade, not crash, when the WordNet corpus is absent.

    ``nltk.corpus.wordnet`` is a lazy loader: it only touches disk on first
    attribute access, so a bare ``try/except`` around the import does not
    catch the ``LookupError`` the corpus raises on that first access. Patching
    ``skos._wordnet`` itself (rather than the ``nltk`` module) simulates that
    outcome without ever touching the real corpus, so this test is meaningful
    even in an environment where WordNet genuinely isn't installed.
    """

    @pytest.fixture(autouse=True)
    def _patch_missing_corpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skos, "_wordnet", lambda: None)

    def test_lexicalize_a_class_survives_without_wordnet(self) -> None:
        concept = lexicalize(
            Concept(uri="cco:Hospital", pref_label="Hospital", definition="", kind="class")
        )
        assert "hospital" in concept.alt_labels

    def test_multiword_label_still_yields_its_content_words(self) -> None:
        concept = lexicalize(
            Concept(
                uri="cco:CommercialOrganization",
                pref_label="Commercial Organization",
                definition="",
                kind="class",
            )
        )
        assert "commercial" in concept.alt_labels
        assert "organization" in concept.alt_labels
        assert "commercial organization" in concept.alt_labels

    def test_wordnet_synonyms_returns_empty_set_instead_of_raising(self) -> None:
        assert _wordnet_synonyms("hospital", "class") == set()


class TestAncestors:
    """``ancestors`` closes the subsumption walk over EVERY declared parent."""

    def test_walks_every_parent_not_only_the_first(self) -> None:
        """Multiple inheritance is normal in CCO; broader[0] hides whole branches.

        ``ancestors`` gates two live decisions — which evoked classes are
        redundant in ``entity_labels``, and which properties are domain/range
        satisfiable in ``relation_labels``. Following one parent understates
        both, so a property declared over the branch that was skipped never
        gets offered to the extractor.
        """
        scheme = {
            "Dance Studio": _cls("Dance Studio", ("Facility", "Commercial Organization")),
            "Facility": _cls("Facility", ("Independent Continuant",)),
            "Commercial Organization": _cls("Commercial Organization", ("Organization",)),
            "Independent Continuant": _cls("Independent Continuant"),
            "Organization": _cls("Organization"),
        }
        assert set(ancestors(scheme, "Dance Studio")) == {
            "Facility",
            "Commercial Organization",
            "Independent Continuant",
            "Organization",
        }

    def test_terminates_on_a_cycle_without_returning_itself(self) -> None:
        """A concept is never its own ancestor — it would prune itself as redundant."""
        scheme = {"A": _cls("A", ("B",)), "B": _cls("B", ("A",))}
        assert ancestors(scheme, "A") == ["B"]

    def test_unknown_label_has_no_ancestors(self) -> None:
        assert ancestors({}, "Nothing") == []

    def test_depth_bounds_the_walk(self) -> None:
        scheme = {"A": _cls("A", ("B",)), "B": _cls("B", ("C",)), "C": _cls("C")}
        assert ancestors(scheme, "A", depth=1) == ["B"]
