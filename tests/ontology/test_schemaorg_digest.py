"""The bundled schema.org digest: the always-on default ontology.

Three things are load-bearing here and each has a test that fails loudly:

1. **No rdflib.** ``rdflib`` ships in the ``assisted`` extra, but the ontology
   layer now runs in the default ``llm_free`` mode a bare ``pip install
   graphknows`` gives. Loading the digest must therefore be pure ``json`` —
   asserted by loading it in a fresh interpreter and checking ``sys.modules``.
2. **Multi-valued domain/range.** schema.org declares ``domainIncludes`` /
   ``rangeIncludes`` and both repeat (``about`` has six domains). A single-string
   field silently kept one of them.
3. **A similarity floor on the label hints.** With 933 classes the top-k is no
   longer "the whole ontology", so an unfloored top-12 injects twelve arbitrary
   labels into every chunk's extraction schema.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.ontology.loader import OntologyTerm, load_ontology_terms, load_rdf_ontology_file

DIGEST = (
    Path(__file__).resolve().parents[2]
    / "graphknows"
    / "ontology"
    / "assets"
    / "schemaorg"
    / "schemaorg.json"
)


@pytest.fixture(scope="module")
def terms() -> list[OntologyTerm]:
    return load_ontology_terms(DIGEST)


def _term(terms: list[OntologyTerm], kind: str, label: str) -> OntologyTerm:
    return next(t for t in terms if t.kind == kind and t.label == label)


class TestDigestLoads:
    def test_load_does_not_import_rdflib(self) -> None:
        """The default ontology path must work without the ``assisted`` extra.

        Run in a fresh interpreter: an ``import rdflib`` anywhere else in the
        test session would make an in-process assertion pass by accident.
        """
        code = (
            "import sys, json;"
            "from graphknows.ontology.loader import load_ontology_terms;"
            f"terms = load_ontology_terms({str(DIGEST)!r});"
            "print(json.dumps({'n': len(terms), 'rdflib': 'rdflib' in sys.modules}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result == {"n": 1878, "rdflib": False}  # 933 classes + 945 properties

    def test_term_counts(self, terms: list[OntologyTerm]) -> None:
        """933 class labels + 945 object properties.

        933, not the 1010 rdfs:Class subjects in the Turtle: the digest is
        namespace-filtered to https://schema.org/. schema.org links other
        vocabularies via owl:equivalentClass and those targets are typed
        rdfs:Class in the same file, so a type-only filter swept in 77 foreign
        terms — SNOMED codes, LRMoo, DCMI, an FIBO BankAccount. They reached a
        live graph (13 unece + 8 SNOMED ONTOLOGY_CLASS nodes in mem_lg30v4) and
        minted ENTITY types like BANKACCOUNT on dialogue.
        """
        assert len([t for t in terms if t.kind == "class"]) == 933
        assert len([t for t in terms if t.kind == "property"]) == 945

    def test_definition_wins_over_a_bare_duplicate_label(self, terms: list[OntologyTerm]) -> None:
        """foaf:Person and schema:Person share a label; the defined one must win."""
        person = _term(terms, "class", "Person")
        assert person.uri == "https://schema.org/Person"
        assert person.definition == "A person (alive, dead, undead, or fictional)."

    def test_single_valued_domain_and_range(self, terms: list[OntologyTerm]) -> None:
        works_for = _term(terms, "property", "worksFor")
        assert works_for.domain == ("Person",)
        assert works_for.range == ("Organization",)
        assert works_for.definition == "Organizations that the person works for."

    def test_multi_valued_domain_keeps_every_declaration(self, terms: list[OntologyTerm]) -> None:
        """``schema:about`` declares six domains; none of them may be dropped."""
        about = _term(terms, "property", "about")
        assert about.domain == (
            "Certification",
            "CommunicateAction",
            "CreativeWork",
            "DefinedTerm",
            "DefinedTermSet",
            "Event",
        )
        assert about.range == ("Thing",)

    def test_label_list_view_also_avoids_rdflib(self) -> None:
        """The DSPy path loads the same digest through ``load_rdf_ontology_file``."""
        manifest = load_rdf_ontology_file(DIGEST)
        assert len(manifest.class_labels) == 933
        assert len(manifest.property_labels) == 945
        assert "Person" in manifest.class_labels
        assert "worksFor" in manifest.property_labels


# ---------------------------------------------------------------------------
# Any-match domain/range validation
# ---------------------------------------------------------------------------


class _StubIndex:
    """The three attributes ``_STMIngestHandler`` reads off an OntologyIndex.

    Not an ``OntologyIndex``: building one embeds every term, and none of these
    assertions are about embeddings — every subject/object type below is an
    exact class label, which ``_map_to_class`` resolves by lookup at score 1.0.
    """

    def __init__(self, classes: list[OntologyTerm], properties: list[OntologyTerm]) -> None:
        self.classes = classes
        self.properties = properties

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]


def _cls(label: str, parents: tuple[str, ...] = ()) -> OntologyTerm:
    return OntologyTerm(uri=f"u:{label}", label=label, definition="", kind="class", parents=parents)


@pytest.fixture
def handler() -> _STMIngestHandler:
    classes = [_cls("Person"), _cls("Organization"), _cls("Place"), _cls("Event")]
    properties = [
        OntologyTerm(
            uri="u:sponsor",
            label="sponsor",
            definition="",
            kind="property",
            domain=("Place", "Organization", "Person"),
            range=("Organization", "Person"),
        ),
        OntologyTerm(
            uri="u:worksFor",
            label="worksFor",
            definition="",
            kind="property",
            domain=("Person",),
            range=("Organization",),
        ),
    ]
    stub = _StubIndex(classes, properties)
    return _STMIngestHandler(
        store=None,
        extractor=None,
        embed_fn=None,
        ontology_index=stub,
        relation_index=stub,
    )


class TestAnyMatchConformance:
    def test_subject_matching_the_second_of_three_domains_is_admitted(
        self, handler: _STMIngestHandler
    ) -> None:
        assert handler._domain_admits("sponsor", "Organization") is True

    def test_subject_matching_no_declared_domain_is_refused(
        self, handler: _STMIngestHandler
    ) -> None:
        assert handler._domain_admits("sponsor", "Event") is False

    def test_object_matching_the_second_of_two_ranges_is_admitted(
        self, handler: _STMIngestHandler
    ) -> None:
        assert handler._range_admits("sponsor", "Person") is True

    def test_person_object_outside_the_range_is_refused(self, handler: _STMIngestHandler) -> None:
        """The one enforced range violation: a Person where the range excludes people."""
        assert handler._range_admits("worksFor", "Person") is False

    def test_non_person_object_outside_the_range_is_still_admitted(
        self, handler: _STMIngestHandler
    ) -> None:
        """The measured narrowing: the ontology is routinely narrower than the data."""
        assert handler._range_admits("worksFor", "Place") is True

    def test_unmappable_type_admits(self, handler: _STMIngestHandler) -> None:
        assert handler._domain_admits("worksFor", "") is True
        assert handler._range_admits("worksFor", "") is True


# ---------------------------------------------------------------------------
# The label-hint similarity floor
# ---------------------------------------------------------------------------


class _FloorIndex:
    """Index stub that honours ``min_sim`` the way ``OntologyIndex`` does."""

    def __init__(self, scores: list[float]) -> None:
        self.classes = [_cls(f"C{i}") for i in range(len(scores))]
        self.properties = [
            OntologyTerm(uri=f"u:p{i}", label=f"p{i}", definition=f"d{i}", kind="property")
            for i in range(len(scores))
        ]
        self._scores = scores
        self.seen_min_sim: list[float] = []

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]

    def _hits(self, terms: list[OntologyTerm], top_k: int, min_sim: float) -> list[Any]:
        self.seen_min_sim.append(min_sim)
        return [(t, s) for t, s in zip(terms, self._scores, strict=True) if s >= min_sim][:top_k]

    def match_classes(self, embedding, top_k=12, min_sim=0.0):  # type: ignore[no-untyped-def]
        return self._hits(self.classes, top_k, min_sim)

    def match_properties(self, embedding, top_k=12, min_sim=0.0):  # type: ignore[no-untyped-def]
        return self._hits(self.properties, top_k, min_sim)


def _handler_for(index: Any) -> _STMIngestHandler:
    return _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )


class TestHintFloor:
    def test_the_two_declarations_of_the_floor_agree(self) -> None:
        """``DEFAULT_MIN_SIM`` is the fallback for direct callers; settings is the knob."""
        from graphknows.ontology.catalog import DEFAULT_MIN_SIM
        from graphknows.settings import GraphKnowsSettings

        assert DEFAULT_MIN_SIM == GraphKnowsSettings().ontology_hint_min_sim == 0.50

    def test_hints_are_floored_not_just_top_k(self) -> None:
        """Only terms above the floor are hinted — top-k alone injects noise.

        Classes only: property hints are no longer selected by cosine at all
        (see ``_relation_labels`` and ``tests/ontology/test_relation_profile``).
        """
        index = _FloorIndex([0.70, 0.60, 0.40, 0.10])  # floor is 0.50
        handler = _handler_for(index)
        assert handler._label_hints([[0.1] * 4]) == [["C0", "C1"]]
        assert index.seen_min_sim == [handler._hint_min_sim]

    def test_a_chunk_matching_nothing_is_logged_not_silently_hinted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        index = _FloorIndex([0.01, 0.01])
        handler = _handler_for(index)
        with caplog.at_level(logging.INFO, logger="graphknows.ingestion.stm.ingest"):
            assert handler._label_hints([[0.1] * 4, [0.1] * 4]) == [[], []]
        assert "2/2" in caplog.text
