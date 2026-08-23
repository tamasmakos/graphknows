"""CURIEs come from the shipped table, never from what a document happens to bind.

Merging a folder of RDF made rdflib invent ``default1``..``default15`` and
``MilitaryRanksOntology1``. Those are not identifiers, so :mod:`prefixes`
resolves every namespace against a shipped table and mints a *stable* fallback
for the rest. The four claims below are the ones that break silently: a
gen-delim split that beats a known root, a vann declaration ignored, a minted
prefix that changes between runs, and a typo'd host read as a new namespace.
"""

from __future__ import annotations

import pytest

from graphknows.symbolic.ontology.rdf.prefixes import BOOTSTRAP, Prefixes, split_iri

CCO2 = "https://www.commoncoreontologies.org/"
MRO = "http://www.ontologylibrary.mil/CommonCore/Mid/MilitaryRanksOntology/"


def test_table_match_beats_the_gen_delim_split() -> None:
    # A namespace is not derivable from an IRI: this one gen-delim-splits into
    # ".../MilitaryRanksOntology/SSD/", a namespace that does not exist.
    pfx = Prefixes()
    assert pfx.split(MRO + "SSD/Colonel") == (MRO, "SSD/Colonel")
    assert split_iri(MRO + "SSD/Colonel") == (MRO + "SSD/", "Colonel")


def test_longest_known_namespace_wins() -> None:
    # /mro/ is nested inside the plain CCO root; the shorter root must not claim it.
    pfx = Prefixes()
    assert pfx.curie(CCO2 + "mro/Rank") == "ccomro:Rank"
    assert pfx.curie(CCO2 + "ont00001760") == "cco:ont00001760"


def test_a_known_namespace_is_never_flagged_unknown() -> None:
    pfx = Prefixes()
    pfx.curie(CCO2 + "ont00001760")
    assert pfx.unknown == set()


def test_unrecognised_namespace_mints_a_stable_hashed_prefix() -> None:
    # Deliberately ugly, and deliberately not order-dependent: a prefix that
    # changed between two ingests of one corpus would fork every CURIE-keyed row.
    first, second = Prefixes(), Prefixes()
    minted = first.curie("http://ex/Dog")
    assert minted.startswith("ex_") and minted.endswith(":Dog")
    assert minted == second.curie("http://ex/Dog")
    assert "http://ex/" in first.unknown


def test_malformed_local_name_is_recorded_not_repaired() -> None:
    pfx = Prefixes()
    pfx.curie("http://ex/bad name")
    assert pfx.defects and "bad name" in pfx.defects[0]


def test_near_duplicates_finds_the_typo_host() -> None:
    # `wwww.ontologylibrary.mil` ships in the corpus next to the real host. It is
    # kept as its own namespace — aliasing would merge different IRIs — so the
    # only defence is reporting it.
    pfx = Prefixes()
    groups = pfx.near_duplicates(set(BOOTSTRAP))
    assert any(any("wwww." in ns for ns in group) for group in groups)


def test_near_duplicates_ignores_namespaces_that_differ_for_real() -> None:
    assert Prefixes().near_duplicates(["http://a.org/", "http://b.org/"]) == []


def test_vann_declaration_is_adopted_as_the_prefix() -> None:
    rdflib = pytest.importorskip("rdflib")
    graph = rdflib.Graph()
    graph.parse(
        data="""
        @prefix vann: <http://purl.org/vocab/vann/> .
        <http://ex.org/vocab> vann:preferredNamespacePrefix "exv" ;
                              vann:preferredNamespaceUri "http://ex.org/vocab#" .
        """,
        format="turtle",
    )
    pfx = Prefixes()

    assert pfx.learn_vann(graph) == 1
    assert pfx.curie("http://ex.org/vocab#Thing") == "exv:Thing"


def test_vann_never_overrides_a_shipped_binding() -> None:
    rdflib = pytest.importorskip("rdflib")
    graph = rdflib.Graph()
    graph.parse(
        data=f"""
        @prefix vann: <http://purl.org/vocab/vann/> .
        <{CCO2}> vann:preferredNamespacePrefix "whatever" .
        """,
        format="turtle",
    )
    pfx = Prefixes()
    pfx.learn_vann(graph)
    assert pfx.curie(CCO2 + "x") == "cco:x"
