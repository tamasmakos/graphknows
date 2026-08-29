"""Lexicalization: what ``alt_labels`` a concept gets, and what upper-ontology
properties are excluded from.

Issue #141 deleted synonym expansion from ``lexicalize`` entirely — a content
word's WordNet expansion is now a MATCH-TIME decision anchored to a mention's
context (``senses.sense_lemmas`` + ``lexical.evoked``), never a build-time one.
This file now covers the two things that survive that change: the
``_is_upper_ontology_property`` blocklist (still applied, now at the expansion
site instead of here — see ``lexical._sense_anchored_matches``) and
``ancestors``, the ``broader`` closure. It also keeps the missing-corpus
regression coverage for ``lexicalize`` itself, which is not really exercising
WordNet at all any more — see ``TestMissingWordnetCorpus``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphknows.symbolic.ontology.skos import (
    Concept,
    _is_upper_ontology_property,
    ancestors,
    content_tokens,
    lexicalize,
    vocabulary_profile,
)

CCO_DIGEST = (
    Path(__file__).resolve().parents[2]
    / "graphknows"
    / "symbolic"
    / "ontology"
    / "assets"
    / "cco"
    / "cco.json"
)
_CCO_PROFILE = vocabulary_profile(json.loads(CCO_DIGEST.read_text(encoding="utf-8")))


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
    """Hand-built ``Concept``s have no scheme behind them (unstamped
    ``upper_ontology`` defaults ``False``), so the real CCO-derived profile
    (``_CCO_PROFILE``, computed above from the shipped digest) is passed in
    explicitly — exactly the "profile the test supplies" seam
    ``_is_upper_ontology_property`` exists for. A hand-written profile would pin
    whatever the test author typed rather than what ``vocabulary_profile``
    actually derives from CCO's own shape.
    """

    _PROFILE = _CCO_PROFILE

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
        assert _is_upper_ontology_property(_prop(label, domain, rng), self._PROFILE)

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
        assert not _is_upper_ontology_property(_prop(label, domain, rng), self._PROFILE)

    def test_classes_are_never_upper_ontology_properties(self) -> None:
        """The guard is about PROPERTIES; a class keeps its synonyms regardless."""
        cls = Concept(uri="cco:X", pref_label="Grocery Store", definition="", kind="class")
        assert not _is_upper_ontology_property(cls, self._PROFILE)


def _cls(label: str, broader: tuple[str, ...] = ()) -> Concept:
    return Concept(
        uri=f"cco:{label}", pref_label=label, definition="", kind="class", broader=broader
    )


class TestLexicalizePropertiesReachTheirBaseForm:
    """Build-time WordNet synonym expansion is gone; base-form reach is not.

    CCO names properties as inflected present-tense verbs ("requires",
    "permits"), but matching runs on spaCy lemmas everywhere else
    (``lexical.lemma_view``) and WordNet's own lemma names are always base
    forms. Without lemmatising the label itself, an inflected-only altLabel
    is an index key that neither the STRONG lexical match nor sense-anchored
    expansion (``lexical._sense_anchored_matches``) could ever reach.
    """

    def test_an_inflected_property_label_also_reaches_its_lemma(self) -> None:
        c = lexicalize(_prop("teaches", ("Person",), ("Person",)))
        assert "teaches" in c.alt_labels
        assert "teach" in c.alt_labels

    def test_a_class_label_is_not_lemmatised(self) -> None:
        """Classes are left alone: their labels are already base-form nouns."""
        c = lexicalize(_cls("Commercial Organization"))
        assert c.alt_labels == {"commercial organization", "commercial", "organization"}


class TestMissingWordnetCorpus:
    """``lexicalize`` no longer touches WordNet at all, so these now hold trivially.

    Kept rather than deleted: they still pin that a class's own label and
    content words populate ``alt_labels`` correctly, with no corpus involved.
    """

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


class TestCamelCaseLabelsSplit:
    """schema.org names properties in camelCase (``worksFor``) where CCO uses
    space-separated Title Case (``Act of Employment``). ``content_tokens``
    lowercases before tokenizing, destroying the case boundary that would
    split a camelCase label — a collapsed token like ``worksfor`` is an index
    key no mention of running text can ever reach.
    """

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("worksFor", ["works"]),
            ("friendOf", ["friend"]),
            ("lifePartnerOf", ["life", "partner"]),
            ("plansTo", ["plans"]),
            ("EducationalOrganization", ["educational", "organization"]),
            ("hasOccupation", ["occupation"]),
        ],
    )
    def test_camel_case_label_splits_into_lowercase_tokens(
        self, label: str, expected: list[str]
    ) -> None:
        # "has" (in "hasOccupation") is structural through the CCO-derived
        # profile (28% of CCO's property labels), not a global constant.
        assert content_tokens(label, _CCO_PROFILE.structural_tokens) == expected

    def test_has_prefix_is_structural(self) -> None:
        """``has`` is a spaCy stopword ``lexical.lemma_view`` never emits, so a
        two-token ``has <noun>`` label (``hasOccupation``, and the ~70 CCO
        relations shaped ``has brother`` / ``has mother``) can never satisfy
        ``lexical.evoked``'s STRONG all-tokens-present rule or the WEAK
        branch's single-token gate. Stripping it collapses these to their one
        real content word, restoring both paths.

        ``has`` is 28% of CCO's property labels, so the CCO-derived profile
        (not a global constant any more) flags it structural.
        """
        assert content_tokens("has brother", _CCO_PROFILE.structural_tokens) == ["brother"]

    def test_cco_title_case_labels_are_unchanged(self) -> None:
        """No-regression half of the fix: CCO has zero camelCase labels.

        ``act`` comes from CCO's own upper-ontology class labels (``Act``'s
        descendants), so it is now derived rather than assumed.
        """
        assert content_tokens("Act of Employment", _CCO_PROFILE.structural_tokens) == ["employment"]
        assert content_tokens("Commercial Organization", _CCO_PROFILE.structural_tokens) == [
            "commercial",
            "organization",
        ]

    def test_camel_case_class_reaches_its_parts_through_lexicalize(self) -> None:
        c = lexicalize(
            Concept(
                uri="schema:EducationalOrganization",
                pref_label="EducationalOrganization",
                definition="",
                kind="class",
            )
        )
        assert "educational" in c.alt_labels
        assert "organization" in c.alt_labels
        assert "educational organization" in c.alt_labels


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
