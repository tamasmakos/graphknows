"""Persist the RDF projection and its taxonomy into an ArcadeDB namespace.

Two differently-keyed layers in ONE database:

* ``ONODE --(RDF_REL)--> ONODE`` — IRI-keyed, faithful to the RDF. One vertex
  per IRI, one edge type carrying the predicate (dynamic predicates live on the
  edge, not in the type name, or the schema grows a type per vocabulary term).
* ``ONTOLOGY_CLASS --(BROADER)--> ONTOLOGY_CLASS`` — label-keyed, hierarchy
  only. This is the package's own ontology vertex, so ``uri`` must be populated:
  ``write_instance_of`` and ``write_class_matches`` both MATCH on it, and a
  taxonomy without it cannot be joined to the knowledge graph at all.

Two decisions worth stating.

**The DDL lives here and is applied by this tool only.** It is deliberately NOT
in ``storage/arcadedb/_schema.py``: ``CREATE INDEX ON ONTOLOGY_CLASS(key)
UNIQUE`` against a pre-existing namespace whose rows have no ``key`` is a
migration hazard, and every namespace that never runs this tool would pay it on
the next connect.

**Everything MERGEs.** The prototype ``CREATE``d its edges, so a second run
doubled them. An RDF import is exactly the operation someone re-runs — after a
source file changes, after a parse fix — so edge identity is spelled out in the
MERGE pattern and re-running converges instead of accumulating.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graphknows.storage.arcadedb.client import ArcadeDBClient
from graphknows.symbolic.ontology.rdf.prefixes import Prefixes
from graphknows.symbolic.ontology.rdf.projection import DOCUMENT, ROLES, NodeMeta, PredInfo
from graphknows.symbolic.ontology.rdf.taxonomy import Taxon

log = logging.getLogger("graphknows.symbolic.ontology.rdf.store")

BATCH = 500

# `alias` is multi-valued and rides in the `aliases` LIST instead.
ROLE_COLS: list[str] = [r for r in ROLES if r != "alias"]

DDL: list[str] = [
    "CREATE VERTEX TYPE ONODE IF NOT EXISTS",
    "CREATE PROPERTY ONODE.iri IF NOT EXISTS STRING",
    "CREATE PROPERTY ONODE.curie IF NOT EXISTS STRING",
    # The matched namespace, as queryable provenance: the CCO corpus ships three
    # typo'd hosts and this is what makes them countable instead of invisible.
    "CREATE PROPERTY ONODE.ns IF NOT EXISTS STRING",
    "CREATE PROPERTY ONODE.deprecated IF NOT EXISTS BOOLEAN",
    "CREATE PROPERTY ONODE.types IF NOT EXISTS LIST",
    "CREATE PROPERTY ONODE.langs IF NOT EXISTS LIST",
    "CREATE PROPERTY ONODE.aliases IF NOT EXISTS LIST",
    "CREATE PROPERTY ONODE.xrefs IF NOT EXISTS LIST",
    "CREATE PROPERTY ONODE.prop_iris IF NOT EXISTS LIST",
    "CREATE PROPERTY ONODE.prop_names IF NOT EXISTS LIST",
    # ArcadeDB Cypher rejects map-valued properties, so every literal rides as
    # JSON text keyed by CURIE. `prop_names` carries each predicate's human name
    # resolved from the graph itself — the only thing that makes cco:ont00001760
    # legible to someone reading a row.
    "CREATE PROPERTY ONODE.props_json IF NOT EXISTS STRING",
    "CREATE PROPERTY ONODE.doc_json IF NOT EXISTS STRING",
    "CREATE PROPERTY ONODE.bnode_meta_json IF NOT EXISTS STRING",
    *[f"CREATE PROPERTY ONODE.{role} IF NOT EXISTS STRING" for role in ROLE_COLS],
    "CREATE INDEX IF NOT EXISTS ON ONODE(iri) UNIQUE",
    "CREATE INDEX IF NOT EXISTS ON ONODE(curie) NOTUNIQUE",
    "CREATE INDEX IF NOT EXISTS ON ONODE(ns) NOTUNIQUE",
    "CREATE EDGE TYPE RDF_REL IF NOT EXISTS",
    "CREATE PROPERTY RDF_REL.predicate IF NOT EXISTS STRING",
    "CREATE PROPERTY RDF_REL.predicate_iri IF NOT EXISTS STRING",
    "CREATE PROPERTY RDF_REL.via IF NOT EXISTS STRING",
    # ONTOLOGY_CLASS and BROADER already exist in the core schema; these are the
    # extra columns the taxonomy layer fills in.
    "CREATE VERTEX TYPE ONTOLOGY_CLASS IF NOT EXISTS",
    "CREATE PROPERTY ONTOLOGY_CLASS.uri IF NOT EXISTS STRING",
    "CREATE PROPERTY ONTOLOGY_CLASS.label IF NOT EXISTS STRING",
    "CREATE PROPERTY ONTOLOGY_CLASS.key IF NOT EXISTS STRING",
    "CREATE PROPERTY ONTOLOGY_CLASS.kind IF NOT EXISTS STRING",
    "CREATE PROPERTY ONTOLOGY_CLASS.definition IF NOT EXISTS STRING",
    "CREATE PROPERTY ONTOLOGY_CLASS.aliases IF NOT EXISTS LIST",
    "CREATE PROPERTY ONTOLOGY_CLASS.curies IF NOT EXISTS LIST",
    "CREATE PROPERTY ONTOLOGY_CLASS.iris IF NOT EXISTS LIST",
    "CREATE PROPERTY ONTOLOGY_CLASS.depth IF NOT EXISTS INTEGER",
    "CREATE PROPERTY ONTOLOGY_CLASS.deprecated IF NOT EXISTS BOOLEAN",
    "CREATE PROPERTY ONTOLOGY_CLASS.merged IF NOT EXISTS INTEGER",
    "CREATE INDEX IF NOT EXISTS ON ONTOLOGY_CLASS(key) UNIQUE",
    "CREATE INDEX IF NOT EXISTS ON ONTOLOGY_CLASS(uri) UNIQUE",
    "CREATE EDGE TYPE BROADER IF NOT EXISTS",
    "CREATE PROPERTY BROADER.via IF NOT EXISTS STRING",
    "CREATE PROPERTY BROADER.bridged IF NOT EXISTS BOOLEAN",
    # Whether this edge may be COMPOSED with the next one, and whether it was
    # asserted or concluded. Without them a traverser cannot tell a subsumption
    # chain it may climb (`rdfs:subClassOf`) from a `skos:broader` link it may
    # not, and every stored hierarchy reads as if it were transitive.
    "CREATE PROPERTY BROADER.transitive IF NOT EXISTS BOOLEAN",
    "CREATE PROPERTY BROADER.inferred IF NOT EXISTS BOOLEAN",
]

NODE_CYPHER = (
    "UNWIND $rows AS row "
    "MERGE (n:ONODE {iri: row.iri}) "
    "SET n.curie = row.curie, n.ns = row.ns, n.deprecated = row.deprecated, "
    "    n.types = row.types, n.langs = row.langs, n.aliases = row.aliases, "
    "    n.xrefs = row.xrefs, n.prop_iris = row.prop_iris, "
    "    n.prop_names = row.prop_names, n.props_json = row.props_json, "
    "    n.doc_json = row.doc_json, n.bnode_meta_json = row.bnode_meta_json, "
    + ", ".join(f"n.{role} = row.{role}" for role in ROLE_COLS)
)

# Edge identity is (subject, object, predicate IRI, via) — two IRIs are commonly
# related by several predicates at once, and the same predicate can arrive both
# asserted and through a class expression. The CURIE is a function of the IRI, so
# it is SET rather than matched on: a prefix-table change must not fork the edge.
EDGE_CYPHER = (
    "UNWIND $rows AS row "
    "MATCH (s:ONODE {iri: row.s}), (o:ONODE {iri: row.o}) "
    "MERGE (s)-[r:RDF_REL {predicate_iri: row.pi, via: row.via}]->(o) "
    "SET r.predicate = row.p"
)

TAXON_CYPHER = (
    "UNWIND $rows AS row "
    "MERGE (t:ONTOLOGY_CLASS {key: row.key}) "
    "SET t.label = row.label, t.kind = row.kind, t.definition = row.definition, "
    "    t.uri = row.uri, t.aliases = row.aliases, t.curies = row.curies, "
    "    t.iris = row.iris, t.depth = row.depth, t.deprecated = row.deprecated, "
    "    t.merged = row.merged"
)

# The taxonomy is a DiGraph, so a (child, parent) pair has exactly one edge:
# `via` and `bridged` describe it, they do not identify it. Merging on them too
# would make a re-run that reclassifies an edge leave the stale one behind.
BROADER_CYPHER = (
    "UNWIND $rows AS row "
    "MATCH (c:ONTOLOGY_CLASS {key: row.c}), (p:ONTOLOGY_CLASS {key: row.p}) "
    "MERGE (c)-[b:BROADER]->(p) "
    "SET b.via = row.via, b.bridged = row.bridged, b.transitive = row.transitive, "
    "    b.inferred = row.inferred"
)


def node_rows(
    meta: dict[str, NodeMeta], pfx: Prefixes, preds: dict[str, PredInfo]
) -> list[dict[str, Any]]:
    """One row per projected IRI. Concept metadata and document metadata stay apart.

    ``props_json`` is keyed by CURIE, which is unambiguous here because the
    prefix table is ours and shipped, so CURIE -> IRI round-trips; ``prop_iris``
    keeps the raw form anyway. Ontology-FILE metadata (license, version,
    creator) goes to ``doc_json``: it is real, but it describes the document, and
    mixing it into a concept's props is what makes the props unreadable.
    """
    rows: list[dict[str, Any]] = []
    for node in meta.values():
        props = {preds[p].curie: dict(v) for p, v in node.props.items() if p not in DOCUMENT}
        doc = {preds[p].curie: dict(v) for p, v in node.props.items() if p in DOCUMENT}
        row: dict[str, Any] = {
            "iri": node.iri,
            "curie": node.curie,
            "ns": pfx.split(node.iri)[0],
            "deprecated": node.deprecated,
            "types": sorted(node.types),
            "langs": sorted(node.langs),
            "aliases": node.aliases,
            "xrefs": sorted(node.xrefs),
            "prop_iris": sorted(node.props),
            "prop_names": sorted({preds[p].name for p in node.props}),
            "props_json": json.dumps(props, sort_keys=True) if props else "",
            "doc_json": json.dumps(doc, sort_keys=True) if doc else "",
            "bnode_meta_json": json.dumps(node.bnode_meta, sort_keys=True)
            if node.bnode_meta
            else "",
        }
        row.update({role: node.role(role) or "" for role in ROLE_COLS})
        rows.append(row)
    return rows


def edge_rows(projected: Any) -> list[dict[str, Any]]:
    """One row per projected edge, carrying the predicate and its provenance."""
    return [
        {"s": s, "o": o, "p": key, "pi": data.get("predicate_iri", ""), "via": data["via"]}
        for s, o, key, data in projected.edges(keys=True, data=True)
    ]


def taxon_rows(taxa: dict[str, Taxon]) -> list[dict[str, Any]]:
    """One row per taxon.

    ``uri`` is ``sorted(iris)[0]``: each IRI belongs to exactly one taxon, so the
    lowest one is stable across runs and unique across the merge. The digest
    export picks the same IRI, which is what lets a digest-derived term join to
    the ONTOLOGY_CLASS row it came from.
    """
    return [
        {
            "key": t.key,
            "label": t.label,
            "kind": t.kind or "unknown",
            "uri": sorted(t.iris)[0],
            "definition": t.definition,
            "aliases": sorted(t.aliases),
            "curies": sorted(t.curies),
            "iris": sorted(t.iris),
            "depth": t.depth,
            "deprecated": t.deprecated,
            "merged": len(t.iris),
        }
        for t in taxa.values()
    ]


def broader_rows(
    taxonomy: Any, inferred: list[tuple[str, str]] | None = None
) -> list[dict[str, Any]]:
    """One row per BROADER edge (child key -> parent key), asserted then inferred.

    ``transitive`` says whether the edge may be composed with the next one:
    ``rdfs:subClassOf`` may, ``skos:broader`` may not. ``inferred`` separates
    what the source SAID from what we concluded — the materialised closure of the
    transitive predicates arrives here as ``inferred`` pairs, so a reader can
    always fall back to the asserted hierarchy alone.

    Args:
        taxonomy: The DAG from :func:`~graphknows.symbolic.ontology.rdf.taxonomy.extract`.
        inferred: ``(child key, parent key)`` closure pairs, already filtered
            against what ``taxonomy`` asserts (see
            :func:`~graphknows.symbolic.ontology.rdf.taxonomy.inferred_pairs`).

    Returns:
        Rows for :data:`BROADER_CYPHER`.
    """
    rows = [
        {
            "c": child,
            "p": parent,
            "via": data["via"],
            "bridged": data["bridged"],
            # Permissive default for a hand-built graph: an unmarked edge reads
            # as transitive, which is what every OWL predicate here is.
            "transitive": data.get("transitive", True),
            "inferred": False,
        }
        for child, parent, data in taxonomy.edges(data=True)
    ]
    rows += [
        {
            "c": child,
            "p": parent,
            "via": "closure",
            "bridged": False,
            "transitive": True,
            "inferred": True,
        }
        for child, parent in inferred or []
    ]
    return rows


async def _batched(
    client: ArcadeDBClient, db: str, cypher: str, rows: list[dict[str, Any]]
) -> None:
    for start in range(0, len(rows), BATCH):
        await client.command(db, cypher, params={"rows": rows[start : start + BATCH]})


async def ensure_schema(client: ArcadeDBClient, db: str, *, reset: bool = False) -> None:
    """Create the database and this tool's schema, idempotently.

    ``reset`` drops the whole database first. An RDF import is all-or-nothing —
    a partial re-import leaves nodes from a source file that no longer exists —
    and dropping is the only way to be sure the namespace holds one corpus.
    """
    if reset:
        await client.drop_database(db)
        log.info("Dropped database %s", db)
    await client.create_database(db)
    for statement in DDL:
        try:
            await client.command(db, statement, language="sql")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                continue
            raise


async def write_projection(
    client: ArcadeDBClient, db: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> None:
    """Write ONODE vertices and then RDF_REL edges (vertices first: edges MATCH)."""
    await _batched(client, db, NODE_CYPHER, nodes)
    await _batched(client, db, EDGE_CYPHER, edges)
    log.info("Wrote %d ONODE / %d RDF_REL to %s", len(nodes), len(edges), db)


async def write_taxonomy(
    client: ArcadeDBClient, db: str, taxa: list[dict[str, Any]], broader: list[dict[str, Any]]
) -> None:
    """Write ONTOLOGY_CLASS vertices and then BROADER edges."""
    await _batched(client, db, TAXON_CYPHER, taxa)
    await _batched(client, db, BROADER_CYPHER, broader)
    log.info("Wrote %d ONTOLOGY_CLASS / %d BROADER to %s", len(taxa), len(broader), db)
