"""Upper-ontology detection must come from the loaded vocabulary's own shape,
not from a hardcoded list of BFO label spellings.

``skos._is_upper_ontology_property`` decides whether an object property is
upper-ontology plumbing -- a property whose domain/range sit so high in the
subsumption DAG that ``labels.relation_labels``'s ancestor walk is satisfied
by any entity at all, so the property wins a budget slot on every chunk while
asserting nothing anyone said. Today that test is a frozenset of 23 hardcoded
BFO label strings (``_BFO_ABSTRACT``): correct for CCO, blind for any other
vocabulary. The fix derives the same judgement structurally: a class counts as
upper-ontology when it sits within 3 levels of a parentless root AND subsumes
at least 5% of the vocabulary's classes (floor 8 descendants).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphknows.symbolic.ontology.skos import (
    Concept,
    _is_upper_ontology_property,
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


def _class(label: str, parents: list[str]) -> dict[str, Any]:
    return {"uri": f"ex:{label}", "label": label, "definition": "", "parents": parents}


def _write_digest(tmp_path: Path) -> dict[str, Any]:
    """A third vocabulary, neither CCO nor the bundled personal profile.

    ``Ens`` is a parentless root with a dozen-odd descendants two to three
    levels deep, plus a leaf pair (``Chef``/``Meal``) far from the root.
    """
    classes = [
        _class("Ens", []),
        _class("Substance", ["Ens"]),
        _class("Quality", ["Ens"]),
        _class("Event", ["Ens"]),
        _class("LivingThing", ["Substance"]),
        _class("Artifact", ["Substance"]),
        _class("Animal", ["LivingThing"]),
        _class("Plant", ["LivingThing"]),
        _class("Tool", ["Artifact"]),
        _class("Vessel", ["Artifact"]),
        _class("Occupation", ["Event"]),
        _class("Person", ["Animal"]),
        _class("Chef", ["Person"]),
        _class("Meal", ["Vessel"]),
    ]
    properties = [
        {
            "uri": "ex:sustains",
            "label": "sustains",
            "definition": "",
            "domain": ["Ens"],
            "range": ["Ens"],
        },
        {
            "uri": "ex:cooks",
            "label": "cooks",
            "definition": "",
            "domain": ["Chef"],
            "range": ["Meal"],
        },
    ]
    digest = {"classes": classes, "properties": properties}
    (tmp_path / "third_vocab.json").write_text(json.dumps(digest), encoding="utf-8")
    return digest


def _property_concept(entry: dict[str, Any]) -> Concept:
    return Concept(
        uri=str(entry["uri"]),
        pref_label=str(entry["label"]),
        definition="",
        kind="property",
        domain=tuple(entry.get("domain") or ()),
        range=tuple(entry.get("range") or ()),
    )


class TestThirdVocabularyGetsUpperOntologyProtection:
    """Neither CCO-spelled nor the personal profile: the digest's own shape
    is the only thing that can tell ``sustains`` (plumbing, typed over the
    root ``Ens``) apart from ``cooks`` (a concrete Chef-cooks-Meal fact)."""

    def test_the_root_typed_property_is_flagged_upper_ontology(self, tmp_path: Path) -> None:
        digest = _write_digest(tmp_path)
        profile = vocabulary_profile(digest)
        sustains = _property_concept(digest["properties"][0])
        assert _is_upper_ontology_property(sustains, profile) is True

    def test_the_concrete_property_is_not_flagged(self, tmp_path: Path) -> None:
        digest = _write_digest(tmp_path)
        profile = vocabulary_profile(digest)
        cooks = _property_concept(digest["properties"][1])
        assert _is_upper_ontology_property(cooks, profile) is False


class TestCcoNoRegression:
    """The class of property measured to produce 64 of 286 REL edges (22%)
    on conv-30 must stay suppressed once the detection is structural."""

    def _cco_digest(self) -> dict[str, Any]:
        return json.loads(CCO_DIGEST.read_text(encoding="utf-8"))

    def _cco_property(self, digest: dict[str, Any], label: str) -> dict[str, Any]:
        return next(p for p in digest["properties"] if p["label"] == label)

    def test_the_digest_has_the_expected_shape(self) -> None:
        digest = self._cco_digest()
        assert len(digest["classes"]) == 1437
        assert len(digest["properties"]) == 264

    def test_the_plumbing_properties_stay_flagged(self) -> None:
        digest = self._cco_digest()
        profile = vocabulary_profile(digest)
        for label in ("permits", "requires", "realizes", "is required by"):
            concept = _property_concept(self._cco_property(digest, label))
            assert _is_upper_ontology_property(concept, profile) is True, label

    def test_everyday_properties_stay_unflagged(self) -> None:
        """``owns``/``teaches``/``works for`` aren't in the bare CCO base --
        they're authored in the personal-profile overlay -- but their real
        domain/range classes (Person, Organization, Facility) are, so the
        structural test is exercised against CCO's own hierarchy shape."""
        digest = self._cco_digest()
        profile = vocabulary_profile(digest)
        candidates = [
            Concept(
                uri="overlay:owns",
                pref_label="owns",
                definition="",
                kind="property",
                domain=("Person",),
                range=("Facility",),
            ),
            Concept(
                uri="overlay:teaches",
                pref_label="teaches",
                definition="",
                kind="property",
                domain=("Person",),
                range=("Act",),
            ),
            Concept(
                uri="overlay:works-for",
                pref_label="works for",
                definition="",
                kind="property",
                domain=("Person",),
                range=("Organization",),
            ),
        ]
        for concept in candidates:
            assert _is_upper_ontology_property(concept, profile) is False, concept.pref_label
