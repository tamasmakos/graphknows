"""Label-keyed taxonomy extraction: every hazard the merge creates, on dict rows.

``extract`` is pure — projection rows in, ``(DiGraph, taxa, stats)`` out — so the
four things that make a lexical taxonomy dangerous are testable without a
database or an RDF parser. Each of them was observed on the CCO corpus: a union
disjunct pointing up instead of down, two IRIs whose shared label turns a real
``subClassOf`` into a self-loop, an unlabelled IRI severing a chain, and a cycle
that no single ontology contains.
"""

from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from graphknows.symbolic.ontology.rdf.taxonomy import extract


def _node(iri: str, label: str, *, types: list[str], **kw) -> dict:
    return {
        "iri": iri,
        "curie": f"v:{iri.split(':')[-1]}",
        "label": label,
        "types": types,
        "aliases": kw.get("aliases", []),
        "definition": kw.get("definition", ""),
    }


@pytest.fixture
def corpus() -> tuple[dict, list[dict]]:
    """Dog/dog are one label in two releases (self-loop); Mystery is unlabelled."""
    nodes = {
        n["iri"]: n
        for n in [
            _node("u:Animal", "Animal", types=["owl:Class"], definition="beast"),
            _node("u:Dog", "Dog", types=["owl:Class"], aliases=["hound"]),
            _node("u:Dog2", "dog", types=["owl:Class"], aliases=["doggo"], definition="canine"),
            _node("u:Mystery", "", types=["owl:Class"]),
            _node("u:Cat", "Cat", types=["owl:Class"]),
            _node("u:owns", "owns", types=["owl:ObjectProperty"]),
            _node("u:has", "has", types=["owl:ObjectProperty"]),
            _node("u:Guide", "Guide Dog", types=["owl:Class"]),
            _node("u:Pet", "Pet", types=["owl:Class"]),
            _node("u:Rex", "Rex", types=["owl:NamedIndividual", "v1:Dog"]),
        ]
    }
    hier = [
        {"s": "u:Dog", "o": "u:Animal", "p": "rdfs:subClassOf", "via": "direct"},
        {"s": "u:Dog2", "o": "u:Dog", "p": "rdfs:subClassOf", "via": "direct"},
        {"s": "u:Cat", "o": "u:Mystery", "p": "rdfs:subClassOf", "via": "direct"},
        {"s": "u:Mystery", "o": "u:Animal", "p": "rdfs:subClassOf", "via": "direct"},
        {"s": "u:owns", "o": "u:has", "p": "rdfs:subPropertyOf", "via": "direct"},
        # defined class: the genus is only reachable through the intersection
        {"s": "u:Guide", "o": "u:Dog", "p": "owl:equivalentClass", "via": "genus"},
        # union disjunct: Dog is NARROWER than Pet, so this edge must be flipped
        {"s": "u:Pet", "o": "u:Dog", "p": "owl:equivalentClass", "via": "disjunct"},
    ]
    return nodes, hier


def test_case_variant_labels_merge_into_one_taxon(corpus) -> None:
    _, taxa, _ = extract(*corpus)
    dog = taxa["dog"]
    assert dog.iris == {"u:Dog", "u:Dog2"}
    assert dog.aliases == {"hound", "doggo"}
    assert dog.definition == "canine"  # an empty definition never wins


def test_named_individuals_are_not_taxa(corpus) -> None:
    _, taxa, stats = extract(*corpus)
    assert "rex" not in taxa
    assert stats["non_taxonomic_skipped"] == 1


def test_genus_from_an_intersection_becomes_a_parent(corpus) -> None:
    graph, _, _ = extract(*corpus)
    assert graph.has_edge("guide dog", "dog")


def test_disjunct_edges_are_flipped_to_child_to_parent(corpus) -> None:
    graph, _, stats = extract(*corpus)
    assert graph.has_edge("dog", "pet") and not graph.has_edge("pet", "dog")
    assert stats["flipped_disjuncts"] == 1


def test_self_loop_from_a_same_label_merge_is_dropped(corpus) -> None:
    graph, _, stats = extract(*corpus)
    assert stats["self_loops_dropped"] == 1
    assert not graph.has_edge("dog", "dog")


def test_chain_bridges_through_an_unlabelled_iri(corpus) -> None:
    graph, taxa, stats = extract(*corpus)
    assert graph["cat"]["animal"]["bridged"] is True
    assert stats["bridged_edges"] == 1
    assert not any("mystery" in key for key in taxa)


def test_depth_is_the_longest_path_from_a_root(corpus) -> None:
    _, taxa, stats = extract(*corpus)
    # Dog has two parents (Animal via subClassOf, Pet via the flipped disjunct).
    assert (taxa["animal"].depth, taxa["pet"].depth) == (0, 0)
    assert (taxa["dog"].depth, taxa["guide dog"].depth) == (1, 2)
    assert sorted(stats["roots"]) == ["animal", "has", "pet"]


def test_kind_comes_from_the_declared_types(corpus) -> None:
    graph, taxa, _ = extract(*corpus)
    assert taxa["owns"].kind == "property" and taxa["dog"].kind == "class"
    assert graph["owns"]["has"]["via"] == "property"


def test_non_hierarchical_edges_are_ignored(corpus) -> None:
    # A restriction filler is a relation, not subsumption: handing the whole
    # edge list over unfiltered must not invent a parent.
    nodes, hier = corpus
    graph, _, _ = extract(
        nodes, [*hier, {"s": "u:Dog", "o": "u:Cat", "p": "u:owns", "via": "restriction"}]
    )
    assert not graph.has_edge("dog", "cat")


def test_a_cycle_is_broken_rather_than_shipped() -> None:
    import networkx as nx

    nodes = {f"u:{x}": _node(f"u:{x}", x, types=["owl:Class"]) for x in "ABC"}
    graph, _, stats = extract(
        nodes,
        [
            {"s": "u:A", "o": "u:B", "p": "rdfs:subClassOf", "via": "direct"},
            {"s": "u:B", "o": "u:C", "p": "rdfs:subClassOf", "via": "direct"},
            {"s": "u:C", "o": "u:A", "p": "rdfs:subClassOf", "via": "direct"},
        ],
    )
    assert len(stats["cycles_broken"]) == 1
    assert nx.is_directed_acyclic_graph(graph)
