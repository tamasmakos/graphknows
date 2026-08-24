"""Loading RDF is offline by construction, and the imports closure is ours to walk.

Three claims, each of which used to rest on a convention rather than on code:

* **rdflib will fetch what you hand it.** ``Graph.parse("http://…")`` performs
  network I/O, and this image has no egress. "We only ever pass local paths" is
  not a guard, so the guard is structural — a ``Path``, opened, with an explicit
  ``format=`` — and these tests hand it the things a guard has to refuse.
* **rdflib follows no ``owl:imports``.** One ``parse`` loads one document, so a
  corpus split across files silently loses whatever the imports define. That
  omission leaves 112 of the 224 properties a CCO-modules-only parse yields
  without a usable domain and range — half the relation vocabulary, gated on
  exactly those.
* **Blank node ids are re-minted on every parse.** Anything keyed on one differs
  run to run, which is why the projection's blank-node blocks carry a skolem id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdflib")

from graphknows.symbolic.ontology import rdf_io
from graphknows.symbolic.ontology.loader import load_ontology_terms, load_rdf_ontology_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ontology_rdf"
IMPORTS = FIXTURES / "imports"
SKOS_CHAIN = FIXTURES / "skos-chain.ttl"

ALPHA = "https://graphknows.io/fixtures/alpha"
BETA = "https://graphknows.io/fixtures/beta"
GAMMA = "https://graphknows.io/fixtures/gamma"


def _labels(graph: object) -> set[str]:
    from rdflib import RDFS

    return {str(o) for o in graph.objects(None, RDFS.label)}  # type: ignore[attr-defined]


class TestOfflineGuard:
    def test_a_url_is_refused_before_any_parser_sees_it(self) -> None:
        with pytest.raises(ValueError, match="local files only"):
            rdf_io.local_path("http://example.org/ontology.ttl")

    @pytest.mark.parametrize("load", [load_ontology_terms, load_rdf_ontology_file])
    def test_no_loader_entry_point_accepts_a_url(self, load) -> None:
        """Both public doors, because a guard on one of them is not a guard."""
        with pytest.raises(ValueError, match="local files only"):
            load("https://example.org/ontology.owl")

    def test_parse_file_refuses_a_string_even_for_a_file_that_exists(self) -> None:
        """The structural half: a string is the only thing rdflib could resolve."""
        with pytest.raises(TypeError, match=r"pathlib\.Path"):
            rdf_io.parse_file(str(SKOS_CHAIN))  # type: ignore[arg-type]

    def test_functional_syntax_names_the_converter(self, tmp_path: Path) -> None:
        """rdflib has no OWL functional-syntax parser at all — say so, and say ROBOT."""
        source = tmp_path / "onto.ofn"
        source.write_text("Prefix(:=<http://example.org/>)\nOntology()", encoding="utf-8")
        with pytest.raises(ValueError, match="ROBOT"):
            rdf_io.parse_file(source)

    def test_an_unknown_suffix_lists_the_formats_that_are_understood(self, tmp_path: Path) -> None:
        source = tmp_path / "notes.md"
        source.write_text("# not an ontology", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported ontology format"):
            rdf_io.parse_file(source)

    @pytest.mark.parametrize("load", [load_ontology_terms, load_rdf_ontology_file])
    def test_an_explicitly_named_unusable_file_raises_rather_than_loading_nothing(
        self, load, tmp_path: Path
    ) -> None:
        """Naming one file and being told "no terms found" hides the actual fix."""
        source = tmp_path / "onto.omn"
        source.write_text("Ontology: <http://example.org/>", encoding="utf-8")
        with pytest.raises(ValueError, match="ROBOT"):
            load(source)

    def test_a_local_file_still_parses(self) -> None:
        assert len(rdf_io.parse_file(SKOS_CHAIN)) > 0


@pytest.fixture(scope="module")
def closure() -> rdf_io.Closure:
    return rdf_io.load_closure([IMPORTS / "a.ttl"])


class TestImportsClosure:
    def test_rdflib_alone_stops_at_one_document(self) -> None:
        """The reason load_closure exists — pinned so nobody 'simplifies' it away."""
        assert "Beta Thing" not in _labels(rdf_io.parse_file(IMPORTS / "a.ttl"))

    def test_a_cycle_resolves_to_every_document_exactly_once(self, closure: rdf_io.Closure) -> None:
        """alpha -> beta -> gamma -> alpha terminates instead of recurring."""
        assert sorted(p.name for p in closure.files) == ["a.ttl", "b.ttl", "c.ttl"]

    def test_the_closure_carries_what_the_imports_define(self, closure: rdf_io.Closure) -> None:
        assert {"Alpha Thing", "Beta Thing", "Gamma Thing"} <= _labels(closure.graph)

    def test_version_iris_are_recorded_for_the_whole_closure(self, closure: rdf_io.Closure) -> None:
        """Which release a digest came from is a property of the closure, not of a file."""
        assert closure.version_iris == {
            ALPHA: f"{ALPHA}/1.0.0",
            BETA: f"{BETA}/2.0.0",
            GAMMA: f"{GAMMA}/3.0.0",
        }

    def test_version_iris_are_sorted_not_in_traversal_order(self) -> None:
        """Two digests of one corpus must not differ by the order the sources came in.

        ``version_iris`` is serialised straight into a digest, and it used to be
        keyed in BFS order — which starts at the ENTRIES. Starting from c.ttl
        walks gamma -> alpha -> beta, so unsorted this file would ship keys in an
        order no other entry point produces, and "is the shipped asset what its
        sources digest to" could never be answered byte-wise.
        """
        from_gamma = rdf_io.load_closure([IMPORTS / "c.ttl"])
        assert list(from_gamma.version_iris) == [ALPHA, BETA, GAMMA]

    def test_an_import_is_resolved_by_its_version_iri_too(self, closure: rdf_io.Closure) -> None:
        """beta imports gamma/3.0.0, and only the parsed catalog knows that is c.ttl."""
        assert closure.unresolved == ()
        assert "Gamma Thing" in _labels(closure.graph)

    def test_the_digest_tool_records_the_closure_it_was_built_from(self) -> None:
        """The point of recording it: a digest that cannot name its release is unauditable."""
        from graphknows.symbolic.ontology.digest import build

        digest = build([IMPORTS / "a.ttl"])
        assert {c["label"] for c in digest["classes"]} == {
            "Alpha Thing",
            "Beta Thing",
            "Gamma Thing",
        }
        assert digest["ontologies"] == {
            ALPHA: f"{ALPHA}/1.0.0",
            BETA: f"{BETA}/2.0.0",
            GAMMA: f"{GAMMA}/3.0.0",
        }
        assert digest["imports_unresolved"] == []

    def test_an_import_with_no_local_file_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        source = tmp_path / "orphan.ttl"
        source.write_text(
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "<http://example.org/orphan> a owl:Ontology ;\n"
            "    owl:imports <http://example.org/nowhere> .\n",
            encoding="utf-8",
        )
        assert rdf_io.load_closure([source]).unresolved == ("http://example.org/nowhere",)


class TestSkolemIds:
    def test_rdflib_bnode_ids_differ_between_parses(self) -> None:
        """The premise. Without this the stability test below proves nothing."""
        first, second = (rdf_io.parse_file(SKOS_CHAIN) for _ in range(2))
        assert set(rdf_io.skolem_ids(first)) != set(rdf_io.skolem_ids(second))

    def test_skolem_ids_are_stable_across_two_parses_of_the_same_file(self) -> None:
        first, second = (rdf_io.parse_file(SKOS_CHAIN) for _ in range(2))
        minted = set(rdf_io.skolem_ids(first).values())
        assert minted and minted == set(rdf_io.skolem_ids(second).values())

    def test_the_projected_blank_node_block_is_identical_across_parses(self) -> None:
        """The consumer: a re-import of an unchanged file must not churn the row."""
        pytest.importorskip("networkx")
        from graphknows.symbolic.ontology.rdf.projection import project

        blocks = []
        for _ in range(2):
            _, meta, _, _ = project(rdf_io.parse_file(SKOS_CHAIN))
            blocks.append(meta["https://graphknows.io/fixtures/places#Java"].bnode_meta)
        assert blocks[0] and blocks[0] == blocks[1]
        assert blocks[0][0]["@id"].startswith("urn:graphknows:skolem:")
