"""A defined class states its genus in ``owl:equivalentClass``, not ``subClassOf``.

CCO writes ~447 classes this way: no ``rdfs:subClassOf`` at all, the parent buried
in ``owl:equivalentClass [ owl:intersectionOf ( <genus> <restriction>... ) ]``.
Reading only ``subClassOf`` leaves every one of them parentless, which truncates
the ancestor walk that relation-offer widening and the BROADER retrieval hop both
stand on. These tests pin the three member kinds apart: a genus is a parent, a
restriction filler is not, and a union member is a CHILD and must never be one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphknows.symbolic.ontology.digest import build

pytest.importorskip("rdflib")


TTL = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/> .

ex:Dog     a owl:Class ; rdfs:label "Dog" .
ex:Person  a owl:Class ; rdfs:label "Person" .
ex:Cat     a owl:Class ; rdfs:label "Cat" .
ex:Pet     a owl:Class ; rdfs:label "Pet" .
ex:assists a owl:ObjectProperty ; rdfs:label "assists" .

# Defined class: genus Dog, plus a restriction that is NOT a parent.
ex:GuideDog a owl:Class ;
    rdfs:label "Guide Dog" ;
    owl:equivalentClass [
        a owl:Class ;
        owl:intersectionOf ( ex:Dog [ a owl:Restriction ;
                                      owl:onProperty ex:assists ;
                                      owl:someValuesFrom ex:Person ] )
    ] .

# Union: Pet is defined AS "Dog or Cat", so those are its CHILDREN.
ex:Pet owl:equivalentClass [ a owl:Class ; owl:unionOf ( ex:Dog ex:Cat ) ] .

# Ordinary asserted subclass, to prove the existing path still works.
ex:Puppy a owl:Class ; rdfs:label "Puppy" ; rdfs:subClassOf ex:Dog .

# Range stated as an anonymous union — the shape `named()` always claimed to
# unwrap but never did, because its blank-node test could not match.
ex:caresFor a owl:ObjectProperty ;
    rdfs:label "cares for" ;
    rdfs:domain ex:Person ;
    rdfs:range [ a owl:Class ; owl:unionOf ( ex:Dog ex:Cat ) ] .
"""


@pytest.fixture
def built(tmp_path: Path) -> dict[str, list[dict]]:
    src = tmp_path / "genus.ttl"
    src.write_text(TTL, encoding="utf-8")
    return build([src])


@pytest.fixture
def digest(built) -> dict[str, list[str]]:
    return {c["label"]: c["parents"] for c in built["classes"]}


def test_a_defined_class_takes_its_genus_from_equivalent_class(digest) -> None:
    assert digest["Guide Dog"] == ["Dog"]


def test_a_restriction_filler_is_a_relation_target_not_a_parent(digest) -> None:
    # "assists some Person" says who the dog helps, not what it is.
    assert "Person" not in digest["Guide Dog"]


def test_a_union_member_is_a_child_so_it_never_becomes_a_parent(digest) -> None:
    # Pet == Dog or Cat means Pet SUBSUMES them; recording them as Pet's parents
    # would invert the edge and put the general term under the specific ones.
    assert digest["Pet"] == []


def test_an_asserted_subclass_still_reads_as_before(digest) -> None:
    assert digest["Puppy"] == ["Dog"]


def test_a_range_stated_as_an_anonymous_union_resolves_to_its_members(built) -> None:
    # Regression guard for a branch that could never execute: `named()` tested
    # for a "_:" prefix that rdflib's BNode never carries, so every union-valued
    # domain and range in CCO read as empty and its property was unofferable.
    prop = next(p for p in built["properties"] if p["label"] == "cares for")
    assert prop["range"] == ["Cat", "Dog"]
    assert prop["domain"] == ["Person"]
