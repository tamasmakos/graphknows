"""Export the extracted taxonomy as a standard graphknows ontology digest.

The digest is the one shape the whole package reads an ontology through
(:func:`~graphknows.symbolic.ontology.loader.load_ontology_terms`), and it is plain JSON:
no rdflib at runtime, no second loader. Emitting one here is what makes an
arbitrary folder of RDF usable by a base install — parse once with the
``assisted`` extra, then point ``GRAPHKNOWS_ONTOLOGY`` at the result.

Two things are baked in at export time rather than left to the reader.

**The canonical URI is ``sorted(taxon.iris)[0]``**, exactly as
:func:`~graphknows.symbolic.ontology.rdf.store.taxon_rows` picks it. A taxon merges every
IRI carrying its label (two CCO releases spell ``Geospatial Region`` two ways),
so the digest and the ``ONTOLOGY_CLASS`` row must agree on which one is the key
or nothing can join them.

**A parent that may not be composed is stamped.** ``skos:broader`` is not
transitive (SKOS keeps ``skos:broaderTransitive`` separate for that), so a digest
that only lists parents leaves a reader unable to tell a chain it may climb from
one it may not. Those parents are listed a second time under
``parents_nontransitive``, and
:func:`~graphknows.symbolic.ontology.skos.ancestors` stops its walk at them.

**Domain and range are inherited down the property hierarchy.** Property
selection is GATED on both (``ingestion/stm/ingest.py::_relation_labels``): a
property declaring neither is never offered to the extractor. 217 of the CCO
corpus's object properties declare neither and inherit both from a parent, so
exporting only the declared constraint silently drops a third of the relation
vocabulary. The digest is consumed by code that cannot walk the hierarchy back
up, which is why this happens here and not at read time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from graphknows.symbolic.ontology.rdf.taxonomy import Taxon

# rdfs:domain / rdfs:range may point into a class expression. A union member and
# an intersection genus are each an acceptable endpoint; a COMPLEMENT is a
# negation ("not a spatial region") and widening on it would assert the opposite
# of what the ontology says.
DR_VIA: frozenset[str] = frozenset({"direct", "genus", "disjunct"})

_ENDPOINT_PREDICATE = {"rdfs:domain": "domain", "rdfs:range": "range"}


def _ancestors(taxonomy: Any, key: str, depth: int = 12) -> set[str]:
    """``key`` plus every BROADER ancestor reachable over TRANSITIVE edges, bounded.

    Keys are casefolded labels throughout — a case mismatch here once made
    domain/range widening inert while every count still looked plausible.

    A non-transitive parent (``skos:broader``) is still an ancestor at one hop
    and is returned, but the walk stops there: ``A broader B broader C`` does not
    assert that A is a C, so inheriting C's domain onto A would be inventing an
    endpoint the source never declared.
    """
    out, frontier = {key}, {key}
    for _ in range(depth):
        frontier = {
            parent
            for child in frontier
            if child in taxonomy
            for parent in taxonomy.successors(child)
            if taxonomy[child][parent].get("transitive", True)
        } - out
        if not frontier:
            break
        out |= frontier
    return out


def _endpoints(
    taxa: dict[str, Taxon], edges: list[dict[str, Any]]
) -> dict[str, dict[str, set[str]]]:
    """Declared ``rdfs:domain`` / ``rdfs:range``, mapped from IRIs onto taxon keys."""
    iri_to_key = {iri: t.key for t in taxa.values() for iri in t.iris}
    tables: dict[str, dict[str, set[str]]] = {
        "domain": defaultdict(set),
        "range": defaultdict(set),
    }
    for edge in edges:
        slot = _ENDPOINT_PREDICATE.get(edge["p"])
        if slot is None or edge["via"] not in DR_VIA:
            continue
        prop_key, class_key = iri_to_key.get(edge["s"]), iri_to_key.get(edge["o"])
        if prop_key and class_key:
            tables[slot][prop_key].add(class_key)
    return tables


def _inherit_endpoints(
    taxonomy: Any, properties: list[str], tables: dict[str, dict[str, set[str]]]
) -> None:
    """Give every undeclared property an ancestor's domain/range.

    An ancestor, not the NEAREST one: ``_ancestors`` returns a flat set with no
    distance attached, so this takes the first declaring ancestor in sorted key
    order. Sorted only for determinism — without it two runs over the same corpus
    hand the same property different endpoints. Picking by hop distance would be
    better and is not what this does; it matters little in practice because the
    gate it feeds enriches rather than filters, and CCO's undeclared properties
    mostly have a single declaring ancestor.
    """
    for key in properties:
        for table in tables.values():
            if table.get(key):
                continue
            for ancestor in sorted(_ancestors(taxonomy, key) - {key}):
                if table.get(ancestor):
                    table[key] = set(table[ancestor])
                    break


def _labels(taxa: dict[str, Taxon], keys: set[str]) -> list[str]:
    return sorted(taxa[k].label for k in keys if k in taxa)


def _module(taxon: Taxon) -> str:
    """The namespace prefix this taxon's CURIEs came from.

    Descriptive only — nothing selects on it. A merged taxon spans two releases
    and therefore two prefixes; the first sorted one is reported so the field
    stays stable rather than order-dependent.
    """
    return sorted(taxon.curies)[0].split(":", 1)[0] if taxon.curies else ""


def to_digest(
    taxonomy: Any, taxa: dict[str, Taxon], edges: list[dict[str, Any]], source: str
) -> dict[str, Any]:
    """Build the digest dict for ``taxa``, in the shape the loader reads.

    Args:
        taxonomy: The ``DiGraph`` from :func:`~graphknows.symbolic.ontology.rdf.taxonomy.extract`
            (child key -> parent key).
        taxa: Its taxa, by key.
        edges: Projection edge rows, read for ``rdfs:domain`` / ``rdfs:range``.
        source: What produced this digest, recorded verbatim.

    Returns:
        ``{"source", "classes", "properties"}`` — the same shape
        :func:`graphknows.symbolic.ontology.digest.build` returns, loadable by
        :func:`~graphknows.symbolic.ontology.loader.load_ontology_terms`.
    """
    tables = _endpoints(taxa, edges)
    property_keys = sorted(k for k, t in taxa.items() if t.kind == "property")
    _inherit_endpoints(taxonomy, property_keys, tables)

    def parents_of(key: str, *, transitive: bool | None = None) -> list[str]:
        if key not in taxonomy:
            return []
        return _labels(
            taxa,
            {
                parent
                for parent in taxonomy.successors(key)
                if transitive is None or taxonomy[key][parent]["transitive"] is transitive
            },
        )

    def entry(key: str) -> dict[str, Any]:
        taxon = taxa[key]
        row: dict[str, Any] = {
            "uri": sorted(taxon.iris)[0],
            "label": taxon.label,
            "definition": taxon.definition,
            "parents": parents_of(key),
            "module": _module(taxon),
        }
        # Stamped only when there is something to stamp, so a digest built from
        # an OWL source is byte-identical to before. Its ABSENCE means every
        # parent came from a transitive predicate (`rdfs:subClassOf`, a defined
        # class's genus, `skos:broaderTransitive`) and may be composed; a name
        # listed here came from `skos:broader`, which may not.
        if nontransitive := parents_of(key, transitive=False):
            row["parents_nontransitive"] = nontransitive
        # `taxon.aliases` is exactly skos:altLabel (and its CCO/OBO synonyms) —
        # `Taxon.absorb` already carries it, this just stops discarding it.
        if alt_labels := sorted(taxon.aliases):
            row["altLabels"] = alt_labels
        # ponytail: skos:example lands in projection.py's "note" role
        # (bundled with scopeNote/IAO_0000112), but Taxon never absorbs it —
        # only definition/aliases/deprecated/types survive the projection ->
        # taxonomy step. No "examples" can be emitted here until that plumbing
        # (projection.NodeMeta -> taxonomy.Taxon) carries the note role too.
        return row

    # Everything that is not a property is a class: `concept` (SKOS) and the
    # untyped remainder are subsumable terms with definitions, and the loader has
    # exactly two kinds. Calling them classes keeps them matchable instead of
    # discarding them for lacking an owl:Class assertion.
    is_property = set(property_keys)
    by_uri = sorted(taxa, key=lambda k: sorted(taxa[k].iris)[0])
    classes = [entry(k) for k in by_uri if k not in is_property]
    properties = [
        {
            **entry(k),
            "domain": _labels(taxa, tables["domain"].get(k, set())),
            "range": _labels(taxa, tables["range"].get(k, set())),
        }
        for k in by_uri
        if k in is_property
    ]
    return {"source": source, "classes": classes, "properties": properties}
