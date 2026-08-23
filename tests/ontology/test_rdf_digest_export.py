"""The exported digest must load back through the standard loader, endpoints and all.

A digest is the only shape a base install can read an ontology through (plain
JSON, no rdflib), so the export is only useful if ``load_ontology_terms``
recovers every field from it. Two things are baked in here rather than left to
the reader: the canonical URI must be the same IRI ``store.taxon_rows`` writes,
or the digest cannot be joined to its ONTOLOGY_CLASS row; and domain/range must
already be inherited, because property selection is gated on both and the reader
cannot walk the hierarchy back up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("networkx")

from graphknows.symbolic.ontology.loader import OntologyTerm, load_ontology_terms
from graphknows.symbolic.ontology.rdf.digest_export import to_digest
from graphknows.symbolic.ontology.rdf.store import taxon_rows
from graphknows.symbolic.ontology.rdf.taxonomy import extract


def _node(iri: str, label: str, kind: str, definition: str = "") -> dict:
    return {
        "iri": iri,
        "curie": f"ex:{label}",
        "label": label,
        "types": [f"owl:{kind}"],
        "aliases": [],
        "definition": definition,
    }


NODES = {
    n["iri"]: n
    for n in [
        _node("http://ex/Person", "Person", "Class", "A human being."),
        _node("http://ex/Animal", "Animal", "Class"),
        _node("http://ex/Dog", "Dog", "Class"),
        _node("http://ex/has", "has", "ObjectProperty", "Generic possession."),
        _node("http://ex/owns", "owns", "ObjectProperty"),
    ]
}

EDGES = [
    {"s": "http://ex/Dog", "o": "http://ex/Animal", "p": "rdfs:subClassOf", "via": "direct"},
    {"s": "http://ex/owns", "o": "http://ex/has", "p": "rdfs:subPropertyOf", "via": "direct"},
    {"s": "http://ex/has", "o": "http://ex/Person", "p": "rdfs:domain", "via": "direct"},
    {"s": "http://ex/has", "o": "http://ex/Animal", "p": "rdfs:range", "via": "direct"},
    # A complement is a negation ("not a Dog"); widening on it asserts the opposite.
    {"s": "http://ex/owns", "o": "http://ex/Dog", "p": "rdfs:domain", "via": "complement"},
]


@pytest.fixture(scope="module")
def digest() -> dict:
    graph, taxa, _ = extract(NODES, EDGES)
    return to_digest(graph, taxa, EDGES, "fixture")


@pytest.fixture(scope="module")
def terms(digest: dict, tmp_path_factory: pytest.TempPathFactory) -> list[OntologyTerm]:
    path = Path(tmp_path_factory.mktemp("digest")) / "fixture.json"
    path.write_text(json.dumps(digest), encoding="utf-8")
    return load_ontology_terms(path)


def _term(terms: list[OntologyTerm], kind: str, label: str) -> OntologyTerm:
    return next(t for t in terms if t.kind == kind and t.label == label)


def test_classes_and_properties_survive_the_round_trip(terms) -> None:
    assert {t.label for t in terms if t.kind == "class"} == {"Person", "Animal", "Dog"}
    assert {t.label for t in terms if t.kind == "property"} == {"has", "owns"}


def test_definitions_and_parents_survive_the_round_trip(terms) -> None:
    assert _term(terms, "class", "Person").definition == "A human being."
    assert _term(terms, "class", "Dog").parents == ("Animal",)
    assert _term(terms, "property", "owns").parents == ("has",)


def test_declared_domain_and_range_survive_the_round_trip(terms) -> None:
    has = _term(terms, "property", "has")
    assert has.domain == ("Person",) and has.range == ("Animal",)


def test_undeclared_property_inherits_its_parents_endpoints(terms) -> None:
    # 217 of the CCO corpus's object properties declare neither, and property
    # selection is gated on both — unexported inheritance means an unofferable
    # third of the relation vocabulary.
    owns = _term(terms, "property", "owns")
    assert owns.domain == ("Person",) and owns.range == ("Animal",)


def test_a_complement_never_supplies_an_endpoint(digest: dict) -> None:
    owns = next(p for p in digest["properties"] if p["label"] == "owns")
    assert "Dog" not in owns["domain"]


def test_uri_matches_the_one_the_store_writes(digest: dict) -> None:
    graph, taxa, _ = extract(NODES, EDGES)
    by_label = {row["label"]: row["uri"] for row in taxon_rows(taxa)}
    exported = {e["label"]: e["uri"] for e in (*digest["classes"], *digest["properties"])}
    assert exported == by_label


def test_source_is_recorded(digest: dict) -> None:
    assert digest["source"] == "fixture"
