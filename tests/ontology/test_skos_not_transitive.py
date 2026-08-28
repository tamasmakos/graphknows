"""``skos:broader`` is not transitive, and every layer has to agree about it.

W3C's own example: Java broader Island broader Landform. A walk that composes
those concludes "Java is a Landform", which the scheme never asserts — SKOS
keeps ``skos:broaderTransitive`` as a separate super-property precisely so the
two can be told apart. ``skos:related`` is disjoint with it and must never
become a hierarchy edge at all.

Until this landed, ``ancestors`` composed everything, and was safe only by
accident: the bundled CCO digest derives every ``broader`` from
``rdfs:subClassOf``, which IS transitive. The fixture here is the first real
SKOS scheme in the repo, and the OWL control arm below is what keeps the fix
from turning into "nothing composes any more".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rdflib")
pytest.importorskip("networkx")

from graphknows.symbolic.ontology import rdf_io
from graphknows.symbolic.ontology.rdf import digest_export, projection, store, taxonomy
from graphknows.symbolic.ontology.skos import ancestors, build_scheme

SKOS_CHAIN = Path(__file__).resolve().parents[1] / "fixtures" / "ontology_rdf" / "skos-chain.ttl"

# The same three-level chain in OWL. Everything the SKOS tests assert must come
# out the other way round here, or the fix has simply disabled subsumption.
OWL_CHAIN = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/> .

ex:Landform a owl:Class ; rdfs:label "Landform" .
ex:Island   a owl:Class ; rdfs:label "Island" ; rdfs:subClassOf ex:Landform .
ex:Java     a owl:Class ; rdfs:label "Java" ; rdfs:subClassOf ex:Island .
"""


class Pipeline:
    """One RDF file taken all the way to a loadable digest, as the tool does it."""

    def __init__(self, source: Path) -> None:
        self.graph = rdf_io.parse_file(source)
        projected, meta, pfx, preds = projection.project(self.graph)
        nodes = store.node_rows(meta, pfx, preds)
        self.edges = store.edge_rows(projected)
        self.hierarchy, self.taxa, self.stats = taxonomy.extract(
            {row["iri"]: row for row in nodes}, self.edges
        )

    @property
    def broader_rows(self) -> list[dict[str, Any]]:
        return store.broader_rows(self.hierarchy, self.inferred)

    @property
    def inferred(self) -> list[tuple[str, str]]:
        return taxonomy.inferred_pairs(
            self.hierarchy, self.taxa, rdf_io.subsumption_closure(self.graph)
        )

    def scheme(self, tmp_path: Path) -> dict[str, Any]:
        digest = digest_export.to_digest(self.hierarchy, self.taxa, self.edges, "test")
        out = tmp_path / "digest.json"
        out.write_text(json.dumps(digest), encoding="utf-8")
        return build_scheme(out)


@pytest.fixture(scope="module")
def skos() -> Pipeline:
    return Pipeline(SKOS_CHAIN)


@pytest.fixture(scope="module")
def owl(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    source = tmp_path_factory.mktemp("owl") / "chain.ttl"
    source.write_text(OWL_CHAIN, encoding="utf-8")
    return Pipeline(source)


class TestTheFixtureIsWhatItClaims:
    def test_the_chain_is_three_levels_deep(self, skos: Pipeline) -> None:
        assert skos.hierarchy.has_edge("java", "island")
        assert skos.hierarchy.has_edge("island", "landform")

    def test_skos_related_never_becomes_a_broader_edge(self, skos: Pipeline) -> None:
        """Disjoint with broaderTransitive: relatedness is not subsumption."""
        assert "coffee" in skos.taxa
        assert not skos.hierarchy.has_edge("java", "coffee")
        assert not skos.hierarchy.has_edge("coffee", "java")


class TestNonTransitiveEdgesAreMarked:
    def test_every_broader_edge_of_a_skos_scheme_is_non_transitive(self, skos: Pipeline) -> None:
        rows = skos.broader_rows
        assert len(rows) == 2
        assert all(row["transitive"] is False for row in rows)
        assert all(row["inferred"] is False for row in rows)

    def test_the_count_is_reported_rather_than_left_to_be_discovered(self, skos: Pipeline) -> None:
        assert skos.stats["non_transitive_edges"] == 2

    def test_an_owl_hierarchy_stays_transitive(self, owl: Pipeline) -> None:
        assert owl.stats["non_transitive_edges"] == 0
        assert all(row["transitive"] is True for row in owl.broader_rows)


class TestMaterialisedClosure:
    def test_a_broader_chain_is_never_composed_into_an_inferred_edge(self, skos: Pipeline) -> None:
        assert rdf_io.subsumption_closure(skos.graph) == []
        assert skos.inferred == []

    def test_a_subclass_chain_is_materialised_and_marked_inferred(self, owl: Pipeline) -> None:
        """The grandparent link is concluded, not asserted, and says so."""
        assert owl.inferred == [("java", "landform")]
        concluded = [row for row in owl.broader_rows if row["inferred"]]
        assert concluded == [
            {
                "c": "java",
                "p": "landform",
                "via": "closure",
                "bridged": False,
                "transitive": True,
                "inferred": True,
            }
        ]


class TestAncestorsStopAtANonTransitiveParent:
    def test_a_three_level_broader_chain_yields_depth_one_ancestors(
        self, skos: Pipeline, tmp_path: Path
    ) -> None:
        """The defect this ticket names: today's walk returned Landform as well."""
        assert ancestors(skos.scheme(tmp_path), "Java") == ["Island"]

    def test_the_digest_stamps_which_parent_may_not_be_composed(
        self, skos: Pipeline, tmp_path: Path
    ) -> None:
        java = skos.scheme(tmp_path)["Java"]
        assert java.broader == ("Island",)
        assert java.broader_nontransitive == {"Island"}

    def test_the_loose_walk_is_still_reachable_when_a_caller_wants_it(
        self, skos: Pipeline, tmp_path: Path
    ) -> None:
        walked = ancestors(skos.scheme(tmp_path), "Java", transitive_only=False)
        assert walked == ["Island", "Landform"]

    def test_a_subclass_chain_still_closes_all_the_way_up(
        self, owl: Pipeline, tmp_path: Path
    ) -> None:
        """The control arm: rdfs:subClassOf IS transitive and must still compose."""
        scheme = owl.scheme(tmp_path)
        assert ancestors(scheme, "Java") == ["Island", "Landform"]
        assert scheme["Java"].broader_nontransitive == frozenset()
