"""The RDF -> property-graph projection, on a fixture that carries every hazard.

The load-bearing claim is the ``via`` split: resolving a class expression to its
leaf IRIs conflates the genus of a defined class, the disjuncts of a union and a
restriction filler, and that conflation is what stranded 447 defined CCO classes
at the top of the taxonomy. The rest of this file pins the other three rules —
blank nodes are never vertices, their literals are still kept, and OWL plumbing
stays out of node metadata.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")
pytest.importorskip("networkx")

from graphknows.symbolic.ontology.rdf.projection import project

CCO = "http://www.ontologyrepository.com/CommonCoreOntologies/"
CCO2 = "https://www.commoncoreontologies.org/"

FIXTURE_TTL = f"""
@prefix : <http://ex/> . @prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix WHATEVER: <{CCO}> .   # deliberately silly document binding
@prefix cco2: <{CCO2}> . @prefix wat: <http://unknown.example/vocab#> .
<{CCO2}ont00009999> rdfs:label "shoe size" .
:Dog a owl:Class ; rdfs:label "Dog"@en, "Hund"@de ; skos:altLabel "hound"@en ;
     WHATEVER:definition "A domestic canine."@en ; wat:vibe "good boy" ;
     <{CCO2}ont00009999> "11" ; owl:deprecated "true"^^xsd:boolean ;
     :age "abc"^^xsd:integer ;
     rdfs:subClassOf :Animal ,
       [ a owl:Restriction ; owl:onProperty :owns ;
         owl:someValuesFrom [ owl:unionOf ( :Bone :Ball ) ] ] .
:Cat a skos:Concept ; skos:prefLabel "Cat"@en ; rdfs:label "Feline"@en ;
     skos:broader :Animal ; rdfs:seeAlso "https://en.wikipedia.org/wiki/Cat" .
# defined class: the genus is INSIDE the intersection, not in subClassOf
:GuideDog owl:equivalentClass
   [ owl:intersectionOf ( :Dog [ a owl:Restriction ; owl:onProperty :owns ;
                                 owl:someValuesFrom :Harness ] ) ] .
:Pet owl:equivalentClass [ owl:unionOf ( :Dog :Cat ) ] .   # disjuncts: NARROWER
[] a owl:Axiom ; owl:annotatedSource :Dog ; owl:annotatedProperty rdfs:label ;
   owl:annotatedTarget "Dog" ; rdfs:comment "named by the vet" .
"""


@pytest.fixture(scope="module")
def projected() -> tuple:
    import rdflib

    graph = rdflib.Graph()
    graph.parse(data=FIXTURE_TTL, format="turtle")
    return project(graph)


def _vias(graph, meta, subject: str) -> set[tuple[str, str]]:
    """``(local name, via)`` for every out-edge of ``subject``."""
    return {
        (meta[o].curie.split(":")[1], data["via"])
        for _, o, data in graph.out_edges(subject, data=True)
    }


class TestClassExpressions:
    def test_genus_inside_an_equivalent_class_intersection_is_via_genus(self, projected) -> None:
        graph, meta, _, _ = projected
        assert ("Dog", "genus") in _vias(graph, meta, "http://ex/GuideDog")

    def test_restriction_filler_beside_that_genus_is_via_restriction(self, projected) -> None:
        # The filler answers "who does it assist", not "what is it" — reading it
        # as a parent is what puts a defined class under a random co-participant.
        graph, meta, _, _ = projected
        assert _vias(graph, meta, "http://ex/GuideDog") == {
            ("Dog", "genus"),
            ("Harness", "restriction"),
        }

    def test_union_members_are_via_disjunct(self, projected) -> None:
        graph, meta, _, _ = projected
        assert _vias(graph, meta, "http://ex/Pet") == {("Dog", "disjunct"), ("Cat", "disjunct")}

    def test_restriction_edge_is_keyed_on_the_restricted_property(self, projected) -> None:
        graph, meta, pfx, _ = projected
        ex = pfx.prefix("http://ex/")
        edges = {(k, meta[o].curie) for _, o, k in graph.out_edges("http://ex/Dog", keys=True)}
        assert (f"{ex}:owns", f"{ex}:Bone") in edges
        assert ("rdfs:subClassOf", f"{ex}:Animal") in edges


class TestBlankNodes:
    def test_blank_nodes_never_become_vertices(self, projected) -> None:
        _, meta, _, _ = projected
        assert all(iri.startswith("http") for iri in meta)

    def test_literals_stranded_on_a_bnode_are_reanchored_on_the_iri(self, projected) -> None:
        # A reified owl:Axiom is not addressable, but what it says is still
        # metadata about :Dog.
        _, meta, _, _ = projected
        blocks = meta["http://ex/Dog"].bnode_meta
        assert blocks and blocks[0]["rdfs:comment"] == "named by the vet"


class TestNodeMetadata:
    def test_document_prefix_bindings_are_ignored(self, projected) -> None:
        # The fixture binds CCO to `WHATEVER:`; the shipped table says `ccov1:`.
        _, _, _, preds = projected
        assert preds[CCO + "definition"].curie == "ccov1:definition"

    def test_opaque_predicate_names_resolve_from_the_graph(self, projected) -> None:
        _, _, _, preds = projected
        assert preds[CCO2 + "ont00009999"].name == "shoe size"

    def test_pref_label_beats_rdfs_label(self, projected) -> None:
        _, meta, _, _ = projected
        assert meta["http://ex/Cat"].label == "Cat"

    def test_language_tagged_labels_are_all_kept(self, projected) -> None:
        _, meta, _, _ = projected
        dog = meta["http://ex/Dog"]
        assert dog.label == "Dog" and dog.role("label", lang="de") == "Hund"

    def test_deprecation_is_a_flag_and_owl_plumbing_never_reaches_props(self, projected) -> None:
        _, meta, _, _ = projected
        dog = meta["http://ex/Dog"]
        assert dog.deprecated is True
        assert not any(p.startswith("http://www.w3.org/2002/07/owl#") for p in dog.props)

    def test_ill_typed_literal_is_kept_lexically_and_flagged(self, projected) -> None:
        _, meta, _, _ = projected
        dog = meta["http://ex/Dog"]
        assert dog.props["http://ex/age"][""] == ["abc"]
        assert dog.ill_typed and "abc" in dog.ill_typed[0]

    def test_unclaimed_predicates_are_kept_for_review_not_dropped(self, projected) -> None:
        _, meta, _, _ = projected
        assert sorted(meta["http://ex/Dog"].unrouted) == [
            "http://ex/age",
            "http://unknown.example/vocab#vibe",
            CCO2 + "ont00009999",
        ]

    def test_iri_valued_literals_become_xrefs(self, projected) -> None:
        _, meta, _, _ = projected
        assert meta["http://ex/Cat"].xrefs == {"https://en.wikipedia.org/wiki/Cat"}
