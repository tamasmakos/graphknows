"""The bundled Common Core Ontologies digest: the always-on default ontology.

Five things are load-bearing here and each has a test that fails loudly:

1. **No rdflib.** ``rdflib`` ships in the ``assisted`` extra, but the ontology
   layer runs in the default ``llm_free`` mode a bare ``pip install graphknows``
   gives. Loading the digest must therefore be pure ``json`` — asserted by
   loading it in a fresh interpreter and checking ``sys.modules``. This is also
   what makes the two memory modes load the ontology identically.
2. **Offerable relations.** Property selection is gated on domain and range, so
   a property declaring neither can never be offered to the extractor. CCO's 87
   familial relations declare neither, and none is inheritable — the whole chain
   up to "has familial relationship to" is empty — so the digest asserts
   Person->Person for that module. Lose that and the conversational relation
   vocabulary silently disappears while everything still passes.
3. **Multi-value-capable domain/range.** CCO happens to declare one domain per
   property, but the field stays a tuple: the any-match conformance below is
   written against multi-valued declarations and a single-string field would
   silently keep one of them.
4. **A similarity floor on the label hints.** With 1437 classes the top-k is not
   "the whole ontology", so an unfloored top-12 injects twelve arbitrary labels
   into every chunk's extraction schema.
5. **The asset is current.** ``cco.json`` is a build PRODUCT checked into git,
   and it went stale for months against a parser fix nobody re-ran. It is now
   rebuilt from its sources and compared, every run — see
   ``TestShippedDigestMatchesItsSources`` for why nothing cheaper works.
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
from graphknows.symbolic.ontology.loader import (
    OntologyTerm,
    load_ontology_terms,
    load_rdf_ontology_file,
)

DIGEST = (
    Path(__file__).resolve().parents[2]
    / "graphknows"
    / "symbolic"
    / "ontology"
    / "assets"
    / "cco"
    / "cco.json"
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
            "from graphknows.symbolic.ontology.loader import load_ontology_terms;"
            f"terms = load_ontology_terms({str(DIGEST)!r});"
            "print(json.dumps({'n': len(terms), 'rdflib': 'rdflib' in sys.modules}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result == {"n": 1701, "rdflib": False}  # 1437 classes + 264 properties

    def test_term_counts(self, terms: list[OntologyTerm]) -> None:
        """1437 classes + 264 object properties, from 11 CCO modules + familial + BFO.

        Object properties only: an ``owl:DatatypeProperty`` describes a literal
        attribute, not a link between two entities, so it can never be an
        ENTITY->ENTITY relation and would only dilute the label space.
        """
        assert len([t for t in terms if t.kind == "class"]) == 1437
        assert len([t for t in terms if t.kind == "property"]) == 264

    def test_a_class_carries_its_definition_and_module(self, terms: list[OntologyTerm]) -> None:
        """The definition is what gets embedded; the module is what makes a match traceable.

        CCO IRIs are opaque (``ont00001262``), so without the module a persisted
        tag cannot be attributed to a source ontology at all.
        """
        person = _term(terms, "class", "Person")
        assert person.uri == "https://www.commoncoreontologies.org/ont00001262"
        assert person.definition == "An Animal that is a member of the species Homo sapiens."
        assert person.module == "AgentOntology"
        assert person.parents == ("Animal",)

    def test_declared_domain_and_range_survive_as_tuples(self, terms: list[OntologyTerm]) -> None:
        supervised = _term(terms, "property", "is supervised by")
        assert supervised.domain == ("Person",)
        assert supervised.range == ("Person",)

    def test_familial_relations_are_offerable(self, terms: list[OntologyTerm]) -> None:
        """The 87 familial properties declare no domain/range, and the gate needs one.

        CCO states none anywhere on the chain ("has brother" -> "is sibling of"
        -> "has familial relationship to" are all empty), so the digest asserts
        Person->Person for the module — true of it by construction. Without
        this every familial relation is unofferable and the conversational
        relation vocabulary is silently empty.
        """
        brother = _term(terms, "property", "has brother")
        assert brother.module == "FamilialRelationsOntology"
        assert (brother.domain, brother.range) == (("Person",), ("Person",))

    def test_the_relation_vocabulary_is_not_mostly_unofferable(
        self, terms: list[OntologyTerm]
    ) -> None:
        """Every property must be selectable, or the BFO resolution regressed.

        264 of 264: BFO supplies the endpoints CCO types against, and unwrapping
        the ``owl:unionOf`` blank nodes supplies the rest. It was 234 while the
        shipped digest predated that unwrapping.
        """
        properties = [t for t in terms if t.kind == "property"]
        offerable = [t for t in properties if t.domain and t.range]
        assert len(offerable) == len(properties) == 264

    def test_a_bfo_typed_constraint_resolves_to_a_label(self, terms: list[OntologyTerm]) -> None:
        """CCO types its relations against BFO, which is an IMPORT.

        ``has process part`` declares ``rdfs:domain obo:BFO_0000015``. Parsing
        the CCO modules without ``cco-imports/bfo-core.ttl`` resolves that IRI to
        nothing and DROPS the constraint — a CCO-modules-only parse leaves 112
        of its 224 properties without both endpoints, silently halving a
        relation vocabulary that is gated on domain and range.
        """
        part_of = _term(terms, "property", "has process part")
        assert (part_of.domain, part_of.range) == (("process",), ("process",))

    def test_a_constraint_is_inherited_from_a_superproperty(
        self, terms: list[OntologyTerm]
    ) -> None:
        """CCO states a constraint once on the general property and specialises below."""
        supervises = _term(terms, "property", "supervises")
        assert supervises.domain and supervises.range

    def test_the_subclass_chain_reaches_bfo(self, terms: list[OntologyTerm]) -> None:
        """Subsumption must not stop at the CCO boundary.

        A property whose domain is ``material entity`` is satisfiable by a
        Person only if Person's parent chain actually reaches it. Emitting BFO's
        classes is what closes that walk; without them ``_class_ancestors``
        degenerates part-way up and the property is never offered.
        """
        by_label = {t.label: t for t in terms if t.kind == "class"}
        chain, current = ["Person"], "Person"
        while parents := by_label.get(current, OntologyTerm("", "", "", "class")).parents:
            current = parents[0]
            chain.append(current)

        assert chain == [
            "Person",
            "Animal",
            "Organism",
            "object",
            "material entity",
            "independent continuant",
            "continuant",
            "entity",
        ]

    def test_label_list_view_also_avoids_rdflib(self) -> None:
        """The DSPy path loads the same digest through ``load_rdf_ontology_file``."""
        manifest = load_rdf_ontology_file(DIGEST)
        assert len(manifest.class_labels) == 1437
        assert len(manifest.property_labels) == 264
        assert "Person" in manifest.class_labels
        assert "has brother" in manifest.property_labels


# ---------------------------------------------------------------------------
# The shipped asset vs. what its sources actually produce
# ---------------------------------------------------------------------------


def _difference(shipped: dict[str, Any], fresh: dict[str, Any]) -> str:
    """One line naming what a fresh build changed, ``""`` when nothing did.

    Not a bare ``assert shipped == fresh``: pytest's rewriting of two 600 KB
    dicts prints a diff nobody reads, and the useful answer is *which term*
    moved.
    """
    for key in ("source", "ontologies", "imports_unresolved", "dropped_expressions"):
        if shipped.get(key) != fresh[key]:
            return f"header {key!r}: shipped {shipped.get(key)!r} != fresh {fresh[key]!r}"
    for kind in ("classes", "properties"):
        old = {t["uri"]: t for t in shipped[kind]}
        new = {t["uri"]: t for t in fresh[kind]}
        if changed := sorted(u for u in old.keys() | new.keys() if old.get(u) != new.get(u)):
            # ponytail: the count plus ONE example. Enough to tell "regenerate
            # it" from "the parser changed under you"; widen to the full list
            # only if a regression ever needs every term named to be diagnosed.
            first = changed[0]
            return (
                f"{len(changed)} of {len(new)} {kind} differ, e.g. {first}:\n"
                f"  shipped {old.get(first)!r}\n  fresh   {new.get(first)!r}"
            )
    return ""


class TestShippedDigestMatchesItsSources:
    """The asset in git must be what ``BUNDLED_SOURCES`` currently digest to.

    This is the check that was missing. ``cco.json`` sat stale for months after a
    blank-node-unwrapping fix in ``digest.py``: 31 properties whose
    ``rdfs:domain``/``rdfs:range`` are ``owl:unionOf`` blank nodes shipped with
    empty endpoint lists, which the offer gate reads as "never offer this
    relation" — 234 offerable instead of 264. Nothing failed, because nothing
    compared the two.

    Two cheaper designs were rejected:

    * **A hash manifest of the .ttl sources.** It would not have caught this at
      all: the sources never changed, the PARSER did.
    * **A marked/opt-in test.** An opt-in drift check is the failure mode, not
      the fix — it is exactly as unrun as no check.

    So it rebuilds, unconditionally, in the subset anyone touching this package
    already runs. Measured cost ~7 s (of which ~5 s is ``rdf_io._catalog``
    parsing every RDF file under the source folders, most of them not in the
    bundle), against a ~55 s ``tests/ontology`` suite. Needs rdflib, which this
    directory already uses; the no-rdflib guarantee is asserted in a SUBPROCESS
    above precisely so an in-process import here cannot weaken it.
    """

    def test_a_fresh_build_reproduces_the_shipped_asset(self) -> None:
        from graphknows.symbolic.ontology.digest import BUNDLED_SOURCES, build

        shipped = json.loads(DIGEST.read_text(encoding="utf-8"))
        assert _difference(shipped, build(BUNDLED_SOURCES)) == "", (
            "cco.json is not what its sources digest to. Regenerate it:\n"
            "  python -m graphknows.symbolic.ontology.digest \\\n"
            "    --source graphknows/symbolic/ontology/assets/cco-modules \\\n"
            "    --source graphknows/symbolic/ontology/assets/cco-extensions"
            "/FamilialRelationsOntology.ttl \\\n"
            "    --source graphknows/symbolic/ontology/assets/cco-imports/bfo-core.ttl \\\n"
            "    --out graphknows/symbolic/ontology/assets/cco/cco.json"
        )


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


def _cls(
    label: str, parents: tuple[str, ...] = (), *, nontransitive: tuple[str, ...] = ()
) -> OntologyTerm:
    return OntologyTerm(
        uri=f"u:{label}",
        label=label,
        definition="",
        kind="class",
        parents=parents,
        parents_nontransitive=nontransitive,
    )


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
        from graphknows.settings import GraphKnowsSettings
        from graphknows.symbolic.ontology.catalog import DEFAULT_MIN_SIM

        assert DEFAULT_MIN_SIM == GraphKnowsSettings().ontology_hint_min_sim == 0.46

    def test_hints_are_floored_not_just_top_k(self) -> None:
        """Only terms above the floor are hinted — top-k alone injects noise.

        Classes only: property hints are no longer selected by cosine at all
        (see ``_relation_labels`` and ``tests/ontology/test_relation_profile``).
        """
        index = _FloorIndex([0.70, 0.60, 0.40, 0.10])  # floor is 0.46
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


# ---------------------------------------------------------------------------
# skos:broader is reported, never composed through
# ---------------------------------------------------------------------------


@pytest.fixture
def skos_handler() -> _STMIngestHandler:
    classes = [
        _cls("Person"),
        _cls("Organization"),
        _cls("Facility", parents=("Organization",)),
        _cls("Dance Studio", nontransitive=("Facility",)),
    ]
    properties = [
        OntologyTerm(
            uri="u:locatedIn",
            label="located in",
            definition="",
            kind="property",
            domain=("Person",),
            range=("Facility",),
        ),
        OntologyTerm(
            uri="u:memberOf",
            label="member of",
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


class TestNonTransitiveParents:
    def test_broader_parent_is_reported_but_not_climbed_past(
        self, skos_handler: _STMIngestHandler
    ) -> None:
        """A skos:broader parent counts as an ancestor; its own parents do not."""
        assert skos_handler._class_ancestors("Dance Studio") == {"dance studio", "facility"}

    def test_subclass_parent_still_composes(self, skos_handler: _STMIngestHandler) -> None:
        """Control: an ordinary rdfs:subClassOf parent still carries its ancestors."""
        assert skos_handler._class_ancestors("Facility") == {"facility", "organization"}

    def test_property_gated_on_the_grandparent_is_not_offered(
        self, skos_handler: _STMIngestHandler
    ) -> None:
        """A chunk evoking only a Dance Studio is offered no Organization relation."""
        entities = [{"type": "Person"}, {"type": "Dance Studio"}]
        assert set(skos_handler._relation_labels(entities)) == {"located in"}
