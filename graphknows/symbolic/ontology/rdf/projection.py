"""Project a folder of RDF files onto a property graph with usable node metadata.

The question this answers is not "how do I load RDF" — rdflib does that — but
what a *property graph* should look like once mixed schemes (OWL / RDFS / SKOS)
and mixed releases of the same ontology have been merged into one graph. Four
rules, each of which exists because the naive reading loses something:

1. **A literal object is metadata, keyed by the FULL predicate IRI + lang tag.**
   CURIEs come from the shipped table in :mod:`~graphknows.symbolic.ontology.rdf.prefixes`
   and never from document bindings — those produced ``default1``..``default15``
   on the CCO corpus.
2. **An IRI object is an edge**, keyed by predicate CURIE plus predicate IRI.
3. **A blank node is never a vertex.** OWL restrictions / unions / intersections
   collapse to direct edges ``S -onProperty-> filler``; literals stranded on a
   blank node (reified ``owl:Axiom``, cardinality restrictions) are re-anchored
   on the IRI they describe rather than dropped, keyed by a SKOLEM id
   (:func:`~graphknows.symbolic.ontology.rdf_io.skolem_ids`) because an rdflib
   ``BNode`` id is different on every parse of the same file.
4. **Every literal predicate is kept, but sorted.** :data:`STRUCTURAL` plumbing
   is dropped (it is already modelled as edges or flags), roles are promoted to
   flat fields, and the rest is kept with a human name resolved from the graph's
   own ``rdfs:label`` — which is the only thing that makes ``cco:ont00001760``
   legible.

The load-bearing part is :func:`_expand`. Resolving a class expression to its
leaf IRIs conflates three different relationships, and that conflation is what
stranded 447 defined CCO classes at the top of the taxonomy: intersection
members are the **genus** (the subject is subsumed by them), union members are
**disjuncts** (they are subsumed by the subject — the direction reverses), and a
restriction filler is not subsumption at all. Each edge therefore carries a
``via`` in ``{direct, genus, disjunct, restriction, complement, enumeration,
expression}``, and the taxonomy layer reads it.

Needs the ``ontology`` extra (rdflib, networkx). The tables below are stdlib.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphknows.symbolic.ontology import rdf_io
from graphknows.symbolic.ontology.rdf.prefixes import Prefixes, split_iri

log = logging.getLogger("graphknows.symbolic.ontology.rdf.projection")

_OWL = "http://www.w3.org/2002/07/owl#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_CCO = "http://www.ontologyrepository.com/CommonCoreOntologies/"
_CCO2 = "https://www.commoncoreontologies.org/"
_OBO = "http://purl.obolibrary.org/obo/"
_DCT = "http://purl.org/dc/terms/"
_DCE = "http://purl.org/dc/elements/1.1/"
_OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"

# Role -> candidate predicate IRIs, highest priority first. Opaque ids resolve
# their own names from the graph (see `index_predicates`), so only the standard
# vocabularies need naming here. Plain strings, not rdflib terms: this table is
# read by `NodeMeta`, which must work in a base install without the extra.
ROLES: dict[str, list[str]] = {
    "label": [_SKOS + "prefLabel", _RDFS + "label", _OBO + "IAO_0000111", _DCT + "title"],
    "alias": [
        _SKOS + "altLabel",
        _CCO + "alternative_label",
        _CCO2 + "ont00001755",
        _OBO + "IAO_0000118",
        _OBOINOWL + "hasExactSynonym",
        _CCO2 + "ont00001753",
    ],
    "definition": [
        _OBO + "IAO_0000115",
        _SKOS + "definition",
        _CCO + "definition",
        _DCT + "description",
    ],
    "source": [_CCO + "definition_source", _CCO2 + "ont00001754", _OBO + "IAO_0000119"],
    "comment": [_RDFS + "comment", _OBO + "IAO_0000116", _SKOS + "editorialNote"],
    "note": [
        _SKOS + "scopeNote",
        _SKOS + "example",
        _OBO + "IAO_0000112",
        _CCO + "example_of_usage",
    ],
    "curated_in": [_CCO + "is_curated_in_ontology", _CCO2 + "ont00001760"],
}
ROLE_OF: dict[str, str] = {iri: role for role, iris in ROLES.items() for iri in iris}

# Literal predicates that are OWL/RDF machinery, not node metadata. Each is
# already represented elsewhere (as an edge, as the `deprecated` flag, or as an
# anchored blank-node block), so copying it into props is pure noise.
STRUCTURAL: frozenset[str] = frozenset(
    {
        _RDF + "type",
        _RDF + "first",
        _RDF + "rest",
        _OWL + "annotatedSource",
        _OWL + "annotatedProperty",
        _OWL + "annotatedTarget",
        _OWL + "deprecated",  # -> NodeMeta.deprecated
        _OWL + "cardinality",
        _OWL + "qualifiedCardinality",
        _OWL + "minCardinality",
        _OWL + "minQualifiedCardinality",
        _OWL + "maxCardinality",
        _OWL + "maxQualifiedCardinality",
        _OWL + "onDatatype",
        _OWL + "withRestrictions",
        _OWL + "members",
    }
)

# Document-level metadata: real, but it describes the ONTOLOGY FILE and not a
# concept. Mixing a license and a modification date into a class's props is what
# makes concept metadata unreadable, so it is kept and carried separately.
DOCUMENT: frozenset[str] = frozenset(
    {
        _OWL + "versionInfo",
        _OWL + "versionIRI",
        _OWL + "priorVersion",
        *(
            _DCT + t
            for t in (
                "title",
                "created",
                "modified",
                "license",
                "rights",
                "creator",
                "contributor",
                "publisher",
                "bibliographicCitation",
            )
        ),
        *(_DCE + t for t in ("identifier", "creator", "contributor", "source", "title", "date")),
    }
)

# Object-position structure, not semantics: an edge to owl:Class or to the
# property a restriction is about says nothing the node's own fields do not.
BORING: frozenset[str] = frozenset({_RDF + "type", _OWL + "onProperty"})

_ANY_URI = "http://www.w3.org/2001/XMLSchema#anyURI"


@dataclass
class NodeMeta:
    """All metadata for one IRI. ``props[predicate_iri][lang] = [values]``."""

    iri: str
    curie: str
    types: set[str] = field(default_factory=set)
    props: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    xrefs: set[str] = field(default_factory=set)  # IRI-valued LITERALS (anyURI)
    bnode_meta: list[dict[str, str]] = field(default_factory=list)
    deprecated: bool = False
    ill_typed: list[str] = field(default_factory=list)  # kept lexically, value=None

    def add(self, pred: str, literal: Any) -> None:
        """Record one literal, flagging it if its datatype does not parse.

        An ill-typed literal (``"abc"^^xsd:integer``) is kept verbatim rather
        than dropped: it is a defect in the source, and silently discarding it
        loses the only evidence of that.
        """
        self.props[pred][literal.language or ""].append(str(literal))
        if literal.datatype is not None and literal.value is None:
            self.ill_typed.append(f"{pred} {literal!r}")
        if str(literal.datatype) == _ANY_URI or str(literal).startswith(("http://", "https://")):
            self.xrefs.add(str(literal))

    def role(self, role: str, lang: str = "en") -> str | None:
        """First value for a role, preferring ``lang``, then untagged, then any."""
        for pred in ROLES[role]:
            by_lang = self.props.get(pred)
            if not by_lang:
                continue
            for want in (lang, "", *by_lang):
                if by_lang.get(want):
                    return by_lang[want][0]
        return None

    def role_all(self, role: str) -> list[str]:
        """Every value for a role, across spellings and language tags."""
        return [v for p in ROLES[role] for vs in self.props.get(p, {}).values() for v in vs]

    @property
    def label(self) -> str | None:
        """Display label, ``skos:prefLabel`` winning over ``rdfs:label``."""
        return self.role("label")

    @property
    def definition(self) -> str | None:
        """Natural-language definition, if any spelling of one is present."""
        return self.role("definition")

    @property
    def aliases(self) -> list[str]:
        """Every alternative label."""
        return self.role_all("alias")

    @property
    def unrouted(self) -> list[str]:
        """Kept metadata that no role claims — the review queue, not a loss."""
        return [p for p in self.props if p not in ROLE_OF and p not in DOCUMENT]

    @property
    def langs(self) -> set[str]:
        """Language tags seen on this node's literals."""
        return {lang for by_lang in self.props.values() for lang in by_lang if lang}

    def __repr__(self) -> str:
        return f"<NodeMeta {self.curie} label={self.label!r} props={len(self.props)}>"


def load_dir(root: Path) -> tuple[Any, list[tuple[str, str, int, str]]]:
    """Parse every parseable RDF file under ``root`` into ONE graph.

    Tolerant on purpose, and the per-file report is why: a real asset folder
    mixes ontologies with import catalogs, digests and junk, and a parse failure
    on one of those must not cost the other 40 files. The report row
    ``(path, format, triples added, status)`` is what makes "the ontology looks
    empty" answerable.
    """
    root = rdf_io.local_path(root)
    graph = rdf_io.new_graph()
    report: list[tuple[str, str, int, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        # The format comes from our own suffix table, never from rdflib's
        # `guess_format`: `parse_file` requires an explicit one, which is what
        # stops a mislabelled document from arriving as a parser error about the
        # wrong syntax — and keeps a URL from ever reaching a parser at all.
        rel, fmt = str(path.relative_to(root)), rdf_io.RDF_FORMATS.get(path.suffix.lower())
        if path.name.startswith(rdf_io.SKIP_NAMES):
            report.append((rel, fmt or "-", 0, "skipped (import catalog)"))
        elif fmt is None:
            report.append((rel, "-", 0, "not rdf (unknown suffix)"))
        else:
            before = len(graph)
            try:
                rdf_io.parse_file(path, graph)
            except Exception as exc:
                report.append((rel, fmt, 0, f"parse failed: {type(exc).__name__}"))
                continue
            added = len(graph) - before
            report.append((rel, fmt, added, "ok" if added else "0 triples (not rdf?)"))
    return graph, report


@dataclass(frozen=True)
class PredInfo:
    """What a metadata predicate IS — resolved from the graph, never guessed."""

    iri: str
    curie: str
    name: str  # rdfs:label from the graph if present, else the local name
    bucket: str  # role | document | other
    role: str | None


def index_predicates(graph: Any, pfx: Prefixes) -> dict[str, PredInfo]:
    """Name and bucket every predicate that carries a literal.

    The names come from the ontology itself: every opaque id in the CCO corpus
    (``cco:ont00001760``, ``obo:IAO_0000602``) declares its own ``rdfs:label``,
    so a hardcoded id table would be both large and permanently out of date.
    """
    from rdflib import Literal
    from rdflib.namespace import RDFS

    out: dict[str, PredInfo] = {}
    for _, p, o in graph:
        if not isinstance(o, Literal) or str(p) in out:
            continue
        iri = str(p)
        label = next((str(x) for x in graph.objects(p, RDFS.label)), "")
        role = ROLE_OF.get(iri)
        bucket = "role" if role else ("document" if iri in DOCUMENT else "other")
        out[iri] = PredInfo(iri, pfx.curie(p), label or split_iri(iri)[1], bucket, role)
    return out


def _collection_items(graph: Any, collection: Any) -> list[Any]:
    """Members of an RDF list, or the node itself when it is not one.

    ``owl:unionOf`` normally points at an ``rdf:first``/``rdf:rest`` chain, but a
    single-member expression is sometimes serialized as the member directly.
    """
    from rdflib.namespace import RDF

    if (collection, RDF.first, None) in graph:
        return list(graph.items(collection))
    return [collection]


def _leaves(graph: Any, node: Any, seen: set[Any] | None = None) -> list[Any]:
    """Resolve a class expression (restriction filler / union / intersection) to IRIs."""
    from rdflib import BNode, URIRef
    from rdflib.namespace import OWL

    if isinstance(node, URIRef):
        return [node]
    seen = seen if seen is not None else set()
    if not isinstance(node, BNode) or node in seen:
        return []
    seen.add(node)
    out: list[Any] = []
    for pred in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue, OWL.onClass):
        for obj in graph.objects(node, pred):
            out += _leaves(graph, obj, seen)
    for pred in (OWL.unionOf, OWL.intersectionOf, OWL.complementOf):
        for coll in graph.objects(node, pred):
            for item in _collection_items(graph, coll):
                out += _leaves(graph, item, seen)
    return out


def _restriction_edges(graph: Any, bnode: Any) -> Iterator[tuple[Any, Any, str]]:
    """``S -onProperty-> filler`` for an OWL restriction.

    The restricted property replaces the predicate that pointed here: "subClassOf
    [ onProperty owns; someValuesFrom Bone ]" asserts that S *owns* a Bone, and
    keeping ``rdfs:subClassOf`` on that edge would make it read as subsumption.
    """
    from rdflib.namespace import OWL

    for prop in graph.objects(bnode, OWL.onProperty):
        for filler in _leaves(graph, bnode):
            yield prop, filler, "restriction"


def _expand(
    graph: Any, pred: Any, bnode: Any, seen: set[Any] | None = None
) -> Iterator[tuple[Any, Any, str]]:
    """Yield ``(predicate, target, via)`` for the class expression hung off ``pred``.

    :func:`_leaves` alone conflates three different relationships, which is what
    stranded 447 defined classes at the top of the taxonomy:

    * members of an INTERSECTION are the genus -> the subject is subsumed BY them
    * members of a UNION are disjuncts -> they are subsumed by the subject, so
      the hierarchy direction reverses
    * a restriction FILLER is a relation target -> not subsumption at all

    The ``via`` this puts on the edge is the only place that distinction
    survives; :data:`~graphknows.symbolic.ontology.rdf.taxonomy.HIER` reads it back.

    ponytail: a nested expression is classified by its OWN operator, with no
    memory of the one enclosing it, so the members of a union sitting inside an
    intersection (``X = Genus AND (A OR B)``) are marked ``disjunct`` and end up
    as children of X — whereas ``X`` only entails ``X ⊑ (A OR B)``, not
    ``A ⊑ X``. Faithful to the prototype these numbers were measured on, and rare
    in the CCO corpus, where an intersection holds a genus and restrictions.
    Fix by threading the enclosing operator through the recursion if a corpus
    ever leans on that shape.
    """
    from rdflib import URIRef
    from rdflib.namespace import OWL

    seen = seen if seen is not None else set()
    if bnode in seen:
        return
    seen.add(bnode)
    if (bnode, OWL.onProperty, None) in graph:
        yield from _restriction_edges(graph, bnode)
        return
    matched = False
    for key, via in (
        (OWL.intersectionOf, "genus"),
        (OWL.unionOf, "disjunct"),
        (OWL.complementOf, "complement"),
        (OWL.oneOf, "enumeration"),
    ):
        for coll in graph.objects(bnode, key):
            matched = True
            for item in _collection_items(graph, coll):
                if isinstance(item, URIRef):
                    yield pred, item, via
                else:
                    yield from _expand(graph, pred, item, seen)
    if not matched:
        for filler in _leaves(graph, bnode):
            yield pred, filler, "expression"


class _Projector:
    """One triple-by-triple pass, with each projection rule its own method.

    A class rather than nested closures for the same reason as
    ``digest.py::_Reader``: the rules are the interesting part and each has to be
    separately readable, whereas the four things they share (graph, prefixes,
    output graph, metadata) are pure plumbing.
    """

    def __init__(self, graph: Any, pfx: Prefixes) -> None:
        import networkx as nx  # type: ignore[import-untyped]

        self.g = graph
        self.pfx = pfx
        self.out = nx.MultiDiGraph()
        self.meta: dict[str, NodeMeta] = {}
        self.dropped: Counter[str] = Counter()

    def node(self, iri: Any) -> NodeMeta:
        """The NodeMeta for an IRI, creating the vertex on first sight."""
        found = self.meta.get(str(iri))
        if found is None:
            found = self.meta[str(iri)] = NodeMeta(str(iri), self.pfx.curie(iri))
            self.out.add_node(str(iri))
        return found

    def edge(self, subject: Any, pred: Any, target: Any, via: str) -> None:
        """Add one predicate-keyed edge, materialising the target.

        The predicate is NOT materialised here: in an asserted triple it is
        vocabulary plumbing (``rdfs:subClassOf``) that no reader wants as a
        vertex. A restriction's ``owl:onProperty`` is the opposite case and its
        caller materialises it explicitly.
        """
        self.node(target)
        self.out.add_edge(
            str(subject),
            str(target),
            key=self.pfx.curie(pred),
            predicate_iri=str(pred),
            via=via,
        )

    def literal(self, subject: NodeMeta, pred: Any, obj: Any) -> None:
        """Route one literal: the deprecation flag, dropped plumbing, or metadata."""
        iri = str(pred)
        if iri == _OWL + "deprecated":
            subject.deprecated = str(obj).lower() in ("true", "1")
        elif iri in STRUCTURAL:
            self.dropped[self.pfx.curie(pred)] += 1
        else:
            subject.add(iri, obj)

    def run(self) -> None:
        """Walk every triple whose subject is an IRI."""
        from rdflib import BNode, Literal, URIRef

        for s, p, o in self.g:
            if isinstance(s, BNode):
                continue  # blank nodes are structure; reached through _leaves instead
            subject = self.node(s)
            if str(p) == _RDF + "type" and isinstance(o, URIRef):
                subject.types.add(self.pfx.curie(o))
            elif isinstance(o, Literal):
                self.literal(subject, p, o)
            elif isinstance(o, URIRef):
                if str(p) not in BORING:
                    self.edge(s, p, o, "direct")
            elif isinstance(o, BNode):
                for pred, target, via in _expand(self.g, p, o):
                    self.node(pred)  # `:owns` may appear nowhere but this restriction
                    self.edge(s, pred, target, via)
        self.out.graph["dropped_structural"] = dict(self.dropped)


def _anchor_bnode_literals(graph: Any, meta: dict[str, NodeMeta], pfx: Prefixes) -> None:
    """Re-attach literals that hang off a blank node to the IRI they describe.

    A reified ``owl:Axiom`` and a cardinality restriction both put real metadata
    on a node that will never be a vertex. Rule 3 says the blank node is not
    addressable — it does not say its content is worthless — so each block is
    anchored on the nearest IRI above it, walking up because restrictions nest
    inside unions.

    Each block carries ``@id``, a SKOLEM id derived from what the blank node
    says. rdflib mints a fresh ``BNode`` id on every parse and iterates them in
    set order, so without one, re-importing an unchanged file produced blocks in
    a different order — a row that churns on every run and no way to say whether
    two runs found the same axiom.
    """
    from rdflib import BNode, Literal, URIRef
    from rdflib.namespace import OWL

    parent: dict[Any, Any] = {}
    for s, _, o in graph:
        if isinstance(o, BNode) and o not in parent:
            parent[o] = s

    def anchor_of(start: Any) -> str | None:
        seen: set[Any] = set()
        cur: Any = start
        while isinstance(cur, BNode) and cur not in seen:
            seen.add(cur)
            cur = parent.get(cur)
        return str(cur) if isinstance(cur, URIRef) else None

    skolem = rdf_io.skolem_ids(graph)
    for bnode in set(graph.subjects()):
        if not isinstance(bnode, BNode):
            continue
        if not any(isinstance(o, Literal) for o in graph.objects(bnode, None)):
            continue
        source = next(iter(graph.objects(bnode, OWL.annotatedSource)), None)
        anchor = str(source) if isinstance(source, URIRef) else anchor_of(bnode)
        if anchor in meta:
            meta[anchor].bnode_meta.append(
                {
                    "@id": skolem[str(bnode)],
                    **{pfx.curie(p): str(o) for p, o in graph.predicate_objects(bnode)},
                }
            )
    for node in meta.values():
        node.bnode_meta.sort(key=lambda block: block["@id"])


def project(graph: Any) -> tuple[Any, dict[str, NodeMeta], Prefixes, dict[str, PredInfo]]:
    """Project an rdflib graph onto ``(MultiDiGraph, meta, prefixes, predicates)``.

    A MultiDiGraph because two IRIs are routinely related by several predicates
    at once (``rdfs:subClassOf`` and ``rdfs:domain``), and collapsing those onto
    one edge loses the predicate that a downstream reader selects on.
    """
    try:
        import networkx  # noqa: F401  — presence check; used by _project_triples
        import rdflib  # noqa: F401  — presence check; used throughout
    except ImportError as exc:
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("Projecting RDF onto a property graph", "ontology") from exc

    pfx = Prefixes()
    if adopted := pfx.learn_vann(graph):
        log.info("Adopted %d vann:preferredNamespacePrefix declarations", adopted)

    projector = _Projector(graph, pfx)
    projector.run()
    projected, meta = projector.out, projector.meta
    _anchor_bnode_literals(graph, meta, pfx)
    for iri, node in meta.items():
        projected.nodes[iri].update(
            curie=node.curie, label=node.label, types=sorted(node.types), meta=node
        )
    return projected, meta, pfx, index_predicates(graph, pfx)
