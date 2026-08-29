"""Coarse entity types must anchor to their curated CCO class (issue #154).

The extractor's guaranteed coarse vocabulary (``labels.COARSE``) is offered on
EVERY chunk. The shipped overlay ``conversational.json`` carries a curated
``coarseAliases`` block mapping each coarse type to the CCO class it MEANS
(``event`` -> ``Act``, ``product`` -> ``Material Artifact``, ``occupation`` ->
``Occupation Role``, ``entity`` -> "" i.e. deliberately untypeable) — but
nothing in the codebase reads that key. ``extend_scheme`` parses only
``classes`` and ``properties``, so the curated anchor is dead data.

``_lexical_candidates`` instead falls back to the SKOS altLabel index, which
``skos.lexicalize`` builds by turning EVERY content word of a class label into
a standalone altLabel. A generic type therefore anchors, at full strength,
onto an arbitrarily specific class that merely CONTAINS that word as a
fragment: ``product`` -> "Product Transport Facility", ``event`` -> "Event
Status Nominal Information Content Entity". This is exactly how 'full-time'
got typed 'Act of Ceremony' in issue #154 — a curated alias must beat an
accidental label fragment, and today it does not.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.stm.ingest import _lexical_candidates
from graphknows.settings import GraphKnowsSettings
from graphknows.symbolic.ontology.catalog import load_ontology_index
from graphknows.symbolic.ontology.labels import COARSE
from graphknows.symbolic.ontology.lexical import usable_index
from graphknows.symbolic.ontology.scheme import load_coarse_aliases, load_scheme

# The overlay's curated coarseAliases block (conversational.json), independent
# of the code under test: what each coarse type is SUPPOSED to anchor to.
CURATED = {
    "person": "Person",
    "organization": "Organization",
    "facility": "Facility",
    "occupation": "Occupation Role",
    "event": "Act",
    "animal": "Animal",
    "product": "Material Artifact",
    "entity": "",  # deliberately untypeable
}


@pytest.fixture(scope="module")
def cfg() -> GraphKnowsSettings:
    return GraphKnowsSettings()


@pytest.fixture(scope="module")
def scheme(cfg: GraphKnowsSettings) -> dict:
    return load_scheme(cfg.ontology_source, cfg.overlay_source)


@pytest.fixture(scope="module")
def lex(scheme: dict) -> dict[str, set[str]]:
    return usable_index(scheme, {}, doc_count=0)


@pytest.fixture(scope="module")
def classes(cfg: GraphKnowsSettings) -> list:
    index = load_ontology_index(cfg.ontology_source)
    assert index is not None, "the bundled CCO digest failed to load"
    return index.classes


@pytest.fixture(scope="module")
def label_index(classes: list) -> dict[str, str]:
    return {t.label.casefold(): t.uri for t in classes if t.kind == "class"}


@pytest.fixture(scope="module")
def coarse_aliases(cfg: GraphKnowsSettings) -> dict[str, str]:
    return load_coarse_aliases(cfg.overlay_source)


def test_curated_coarse_types_are_covered_by_the_fixture() -> None:
    """CURATED must track ``labels.COARSE`` so this file can't drift silently."""
    assert set(COARSE) <= set(CURATED)


@pytest.mark.parametrize("coarse_type", [c for c in COARSE if CURATED[c]])
def test_coarse_type_anchors_to_its_curated_class(
    coarse_type: str,
    classes: list,
    label_index: dict[str, str],
    lex: dict[str, set[str]],
    scheme: dict,
    coarse_aliases: dict[str, str],
) -> None:
    candidates = _lexical_candidates(coarse_type, classes, label_index, lex, scheme, coarse_aliases)
    expected = CURATED[coarse_type]

    assert set(candidates) == {expected}, (
        f"'{coarse_type}' must anchor exactly to the curated class {expected!r}, "
        f"got {sorted(candidates)} instead"
    )


def test_entity_is_deliberately_untypeable(
    classes: list,
    label_index: dict[str, str],
    lex: dict[str, set[str]],
    scheme: dict,
    coarse_aliases: dict[str, str],
) -> None:
    """The overlay maps 'entity' to the empty string on purpose: abstain."""
    candidates = _lexical_candidates("entity", classes, label_index, lex, scheme, coarse_aliases)

    assert candidates == {}, (
        f"'entity' has no curated class (overlay maps it to ''), must abstain, "
        f"got {sorted(candidates)}"
    )


def test_type_with_no_lexical_evidence_abstains(
    classes: list,
    label_index: dict[str, str],
    lex: dict[str, set[str]],
    scheme: dict,
    coarse_aliases: dict[str, str],
) -> None:
    """'full-time' is issue #154's own reproduction case: no coarse alias, no
    exact label, no altLabel fragment should exist for it either.
    """
    candidates = _lexical_candidates("full-time", classes, label_index, lex, scheme, coarse_aliases)

    assert candidates == {}, (
        f"'full-time' has no lexical evidence and must abstain, got {sorted(candidates)}"
    )


def test_person_does_not_leak_into_sibling_person_subtypes(
    classes: list,
    label_index: dict[str, str],
    lex: dict[str, set[str]],
    scheme: dict,
    coarse_aliases: dict[str, str],
) -> None:
    """Today the altLabel fragment index also offers 'Neutral Person', 'Enemy
    Person' and 'Allied Person' for the bare word 'person' — those are more
    specific classes that merely CONTAIN the word, not what 'person' means.
    """
    candidates = _lexical_candidates("person", classes, label_index, lex, scheme, coarse_aliases)

    assert set(candidates) == {"Person"}, (
        f"'person' must anchor exactly to 'Person', not its subtypes, got {sorted(candidates)}"
    )


@pytest.mark.parametrize(
    ("fragment_type", "overreaching_label"),
    [
        ("facility", "Product Transport Facility"),
        ("store", "Grocery Store"),
    ],
)
def test_fragment_never_anchors_a_label_naming_more_than_it_does(
    fragment_type: str,
    overreaching_label: str,
    classes: list,
    label_index: dict[str, str],
    lex: dict[str, set[str]],
    scheme: dict,
    coarse_aliases: dict[str, str],
) -> None:
    """A class label names more concepts than a single word does: the fragment
    must never anchor to a label that names something the type string never
    said (issue #154's 'facility' -> 'Product Transport Facility').
    """
    candidates = _lexical_candidates(
        fragment_type, classes, label_index, lex, scheme, coarse_aliases
    )

    assert overreaching_label not in candidates, (
        f"'{fragment_type}' must not anchor to {overreaching_label!r}, which names "
        f"more than the bare word '{fragment_type}' does; got {sorted(candidates)}"
    )
