"""Derive a label-keyed taxonomy from the IRI-keyed projection. Pure.

Two layers coexist deliberately. The projection is faithful to the RDF: one node
per IRI, every predicate kept. This one is lexical: one node per *label*, edges
only where one term subsumes another.

Why label-keyed at all — the CCO corpus ships two releases, so ``Geospatial
Region`` exists as both ``cco:ont00000472`` and ``ccov1:GeospatialRegion``.
Keying on the casefolded label merges them, which is what a lexical taxonomy is
FOR. It is also what makes self-loops and cycles possible (two IRIs sharing a
label with a real ``subClassOf`` between them becomes an edge from a node to
itself), so both are detected here rather than assumed away.

Nothing in this module touches a database or rdflib: rows in, ``(DiGraph, taxa,
stats)`` out. Reading the projection back out of storage and re-deriving the
taxonomy from it is then a property of the *caller*, not a coupling of this
code — and the tests can exercise every hazard on a dict literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# (predicate, via) -> (direction, kind). Keyed on BOTH because the predicate
# alone is not enough: `owl:equivalentClass` is subsumption when it points at a
# member of an INTERSECTION (the genus of a defined class — 447 classes in the
# CCO corpus have no other parent) and REVERSED subsumption when it points at a
# member of a UNION. Restriction fillers are relations, so they appear nowhere
# here; an edge whose (predicate, via) pair is absent is not hierarchical.
HIER: dict[tuple[str, str], tuple[str, str]] = {
    ("rdfs:subClassOf", "direct"): ("up", "class"),
    ("rdfs:subClassOf", "genus"): ("up", "class"),
    ("rdfs:subClassOf", "disjunct"): ("down", "class"),
    ("owl:equivalentClass", "genus"): ("up", "class"),
    ("owl:equivalentClass", "disjunct"): ("down", "class"),
    ("rdfs:subPropertyOf", "direct"): ("up", "property"),
    ("skos:broader", "direct"): ("up", "concept"),
    ("skos:broaderTransitive", "direct"): ("up", "concept"),
    ("skos:narrower", "direct"): ("down", "concept"),
}

# Not taxonomy members. An individual is instance-of, not subsumption (its class
# membership survives on the projection's `types`), and an `owl:Ontology` node is
# the DOCUMENT — it sat at the top of the tree next to `entity` purely because it
# carries an rdfs:label.
NON_TAXONOMIC: frozenset[str] = frozenset({"owl:NamedIndividual", "owl:Ontology"})


@dataclass
class Taxon:
    """One node of the taxonomy: a label, plus every IRI that carries it."""

    key: str  # casefolded label — the merge key
    label: str  # display form (first-seen casing)
    kind: str = ""
    definition: str = ""
    aliases: set[str] = field(default_factory=set)
    iris: set[str] = field(default_factory=set)
    curies: set[str] = field(default_factory=set)
    deprecated: bool = True  # AND across merged IRIs: live if any source is live
    depth: int = 0

    def absorb(self, node: dict[str, Any]) -> None:
        """Merge one projection row into this taxon.

        The first non-empty definition wins rather than the last: re-running the
        merge in a different file order must not change what the taxon says.
        """
        self.iris.add(node["iri"])
        self.curies.add(node["curie"])
        self.aliases.update(node.get("aliases") or [])
        self.definition = self.definition or (node.get("definition") or "")
        self.deprecated = self.deprecated and bool(node.get("deprecated"))
        types = node.get("types") or []
        self.kind = self.kind or (
            "property"
            if any("Property" in t for t in types)
            else "class"
            if any("Class" in t for t in types)
            else "concept"
            if any("Concept" in t for t in types)
            else ""
        )


def _is_non_taxonomic(node: dict[str, Any]) -> bool:
    types = node.get("types") or []
    return bool(NON_TAXONOMIC & set(types)) and not any("Class" in t for t in types)


def _key_of(nodes: dict[str, dict[str, Any]], iri: str) -> str | None:
    """The merge key for an IRI: its casefolded label, or ``None`` if unlabelled."""
    label = (nodes.get(iri, {}).get("label") or "").strip()
    return label.casefold() or None


def _merge_by_label(nodes: dict[str, dict[str, Any]]) -> dict[str, Taxon]:
    taxa: dict[str, Taxon] = {}
    for node in nodes.values():
        key = _key_of(nodes, node["iri"])
        if key is None:
            continue
        taxon = taxa.get(key)
        if taxon is None:
            taxon = taxa[key] = Taxon(key=key, label=(node["label"] or "").strip())
        taxon.absorb(node)
    return taxa


def _parents_of(hier: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Index child IRI -> (parent IRI, kind), normalising direction exactly once.

    A union disjunct points the wrong way. Flipping it here and nowhere else is
    what keeps every later stage able to assume child -> parent.
    """
    parents: dict[str, list[tuple[str, str]]] = {}
    flipped = 0
    for edge in hier:
        entry = HIER.get((edge["p"], edge["via"]))
        if entry is None:
            continue  # not a hierarchical predicate: a relation, not subsumption
        direction, kind = entry
        child, parent = (edge["s"], edge["o"]) if direction == "up" else (edge["o"], edge["s"])
        flipped += direction == "down"
        parents.setdefault(child, []).append((parent, kind))
    return parents, flipped


def _labelled_ancestors(
    nodes: dict[str, dict[str, Any]],
    parents_of: dict[str, list[tuple[str, str]]],
    iri: str,
    seen: set[str] | None = None,
) -> list[tuple[str, str, bool]]:
    """Nearest LABELLED parents of an IRI, as ``(parent, kind, was_bridged)``.

    An unlabelled IRI would sever a chain (child -> <unlabelled> -> parent), and
    dropping the link loses a real subsumption the corpus asserts. Climb past it
    instead, marking the resulting edge bridged so the shortcut stays auditable.
    """
    seen = seen if seen is not None else set()
    out: list[tuple[str, str, bool]] = []
    for parent, kind in parents_of.get(iri, []):
        if parent in seen:
            continue
        seen.add(parent)
        if _key_of(nodes, parent):
            out.append((parent, kind, False))
        else:
            out += [
                (a, k, True) for a, k, _ in _labelled_ancestors(nodes, parents_of, parent, seen)
            ]
    return out


def _link(
    nodes: dict[str, dict[str, Any]],
    taxa: dict[str, Taxon],
    parents_of: dict[str, list[tuple[str, str]]],
) -> tuple[Any, int, int]:
    """Build the label-keyed DAG, returning ``(graph, self_loops, bridged)``."""
    import networkx as nx  # type: ignore[import-untyped]

    graph = nx.DiGraph()
    for key, taxon in taxa.items():
        graph.add_node(key, taxon=taxon)

    self_loops = bridged = 0
    for iri in list(parents_of):
        child_key = _key_of(nodes, iri)
        if child_key is None:
            continue  # its children were bridged past it
        for parent, kind, was_bridged in _labelled_ancestors(nodes, parents_of, iri):
            parent_key = _key_of(nodes, parent)
            if parent_key == child_key:
                # Two IRIs sharing one label, one subClassOf the other: the merge
                # turned real subsumption into a self-loop. Drop, don't keep.
                self_loops += 1
                continue
            bridged += was_bridged
            if graph.has_edge(child_key, parent_key):
                graph[child_key][parent_key]["bridged"] &= was_bridged
            else:
                graph.add_edge(child_key, parent_key, via=kind, bridged=was_bridged)
    return graph, self_loops, bridged


def _break_cycles(graph: Any) -> list[list[str]]:
    """Cut the minimum number of edges that makes the graph acyclic.

    Merging by label can close a loop that no single ontology contains. Shipping
    it would give every traversal a place to spin, so one edge per cycle is
    removed and the cycle is reported rather than silently tolerated.
    """
    import networkx as nx

    cycles: list[list[str]] = []
    while not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        cycles.append([e[0] for e in cycle])
        graph.remove_edge(cycle[0][0], cycle[0][1])
    return cycles


def extract(
    nodes: dict[str, dict[str, Any]], hier: list[dict[str, Any]]
) -> tuple[Any, dict[str, Taxon], dict[str, Any]]:
    """Build the label-keyed taxonomy from projection rows.

    Args:
        nodes: IRI -> projection row (``iri``, ``curie``, ``label``,
            ``definition``, ``aliases``, ``types``, ``deprecated``).
        hier: Edge rows (``s``, ``o``, ``p``, ``via``). Rows whose
            ``(p, via)`` pair is not in :data:`HIER` are ignored, so the whole
            edge list can be handed over unfiltered.

    Returns:
        ``(graph, taxa, stats)`` — a ``networkx.DiGraph`` of child -> parent
        keys, the merged taxa by key, and the hazard counts every one of these
        rules exists for (flipped disjuncts, dropped self-loops, bridged edges,
        broken cycles).
    """
    try:
        import networkx as nx
    except ImportError as exc:
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("Extracting an RDF taxonomy", "assisted") from exc

    kept = {iri: n for iri, n in nodes.items() if not _is_non_taxonomic(n)}
    stats: dict[str, Any] = {"non_taxonomic_skipped": len(nodes) - len(kept)}

    taxa = _merge_by_label(kept)
    parents_of, stats["flipped_disjuncts"] = _parents_of(hier)
    graph, stats["self_loops_dropped"], stats["bridged_edges"] = _link(kept, taxa, parents_of)
    stats["cycles_broken"] = _break_cycles(graph)

    # depth = longest path from a root (a taxon with no parents), which is only
    # well-defined once the cycles above are gone.
    for key in reversed(list(nx.topological_sort(graph))):
        parents = list(graph.successors(key))
        graph.nodes[key]["taxon"].depth = 1 + max((taxa[p].depth for p in parents), default=-1)

    stats["roots"] = [k for k in graph if graph.out_degree(k) == 0]
    stats["multi_parent"] = sum(1 for k in graph if graph.out_degree(k) > 1)
    return graph, taxa, stats
