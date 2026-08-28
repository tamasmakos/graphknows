"""Functional-property flag on ontology terms (issue #140, B10).

A functional property (owl:FunctionalProperty) admits at most one value per
subject: a person has exactly one ``spouse``, but may ``know`` many people.
Storage needs this flag to invalidate a stale edge instead of coexisting with
it, so ``OntologyTerm`` must carry it. Pure-function fixture test: no mocks,
no network, no model loading.
"""

from __future__ import annotations

from pathlib import Path

import graphknows.symbolic.ontology.loader as loader_module
from graphknows.symbolic.ontology.loader import load_ontology_terms

# Locate the bundled asset relative to the installed package, not the cwd.
_PROFILE = Path(loader_module.__file__).parent / "assets" / "personal" / "personal-profile.json"


def _term(label: str):
    terms = load_ontology_terms(_PROFILE)
    matches = [t for t in terms if t.label == label]
    assert matches, f"no term labelled {label!r} in {_PROFILE}"
    return matches[0]


def test_spouse_is_functional() -> None:
    assert _term("spouse").functional is True


def test_knows_is_not_functional() -> None:
    assert _term("knows").functional is False
