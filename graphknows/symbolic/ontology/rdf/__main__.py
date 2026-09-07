r"""Load a folder of RDF files into an ArcadeDB namespace, and optionally digest it.

One pass over the sources produces both layers: the faithful IRI graph
(``ONODE``/``RDF_REL``) and the label-keyed taxonomy (``ONTOLOGY_CLASS``/
``BROADER``) derived from it. The taxonomy is built from the rows about to be
written rather than read back out of the database, so the two layers cannot
disagree and the tool works the same whether or not the namespace already exists.

Usage::

    python -m graphknows.symbolic.ontology.rdf --source ./my-ontology --ns my_rdf --reset
    python -m graphknows.symbolic.ontology.rdf --source ./my-ontology --ns my_rdf \\
        --digest-out my.json
    export GRAPHKNOWS_ONTOLOGY=my.json

Needs the ``assisted`` extra (rdflib, networkx).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from graphknows.settings import get_settings
from graphknows.storage.arcadedb.client import ArcadeDBClient
from graphknows.storage.namespace import db_name
from graphknows.symbolic.ontology import rdf_io
from graphknows.symbolic.ontology.rdf import digest_export, projection, store, taxonomy


async def _run(source: Path, ns: str, digest_out: Path | None, reset: bool) -> None:
    graph, files = projection.load_dir(source)
    parsed = sum(1 for f in files if f[3] == "ok")
    projected, meta, pfx, preds = projection.project(graph)
    print(
        f"{parsed}/{len(files)} files parsed -> {len(graph)} triples -> "
        f"{len(meta)} nodes / {projected.number_of_edges()} edges"
    )
    for group in pfx.near_duplicates(set(pfx.table) | {pfx.split(m)[0] for m in meta}):
        print(f"   !! near-duplicate namespaces (typo?): {group}")

    nodes = store.node_rows(meta, pfx, preds)
    edges = store.edge_rows(projected)
    hierarchy, taxa, stats = taxonomy.extract({r["iri"]: r for r in nodes}, edges)
    # The closure of the predicates that ARE transitive, materialised once here
    # rather than walked at query time — and marked `inferred`, so what the
    # source asserted stays separable from what we concluded. It comes from a
    # SPARQL property path over the parsed graph, not from the taxonomy DAG:
    # that also holds `skos:broader` edges, which must never be composed.
    inferred = taxonomy.inferred_pairs(hierarchy, taxa, rdf_io.subsumption_closure(graph))
    print(
        f"taxonomy: {len(taxa)} taxa, {hierarchy.number_of_edges()} BROADER "
        f"({stats['flipped_disjuncts']} disjuncts flipped, "
        f"{stats['self_loops_dropped']} self-loops dropped, "
        f"{stats['bridged_edges']} bridged, {len(stats['cycles_broken'])} cycles broken, "
        f"{stats['non_transitive_edges']} non-transitive, "
        f"{len(stats['roots'])} roots) + {len(inferred)} inferred"
    )

    db = db_name(ns)
    settings = get_settings()
    client = ArcadeDBClient(
        settings.arcadedb_url,
        settings.arcadedb_user,
        settings.arcadedb_password.get_secret_value(),
    )
    await client.connect()
    try:
        await store.ensure_schema(client, db, reset=reset)
        await store.write_projection(client, db, nodes, edges)
        await store.write_taxonomy(
            client, db, store.taxon_rows(taxa), store.broader_rows(hierarchy, inferred)
        )
    finally:
        await client.close()
    print(f"wrote {len(nodes)} ONODE / {len(edges)} RDF_REL / {len(taxa)} ONTOLOGY_CLASS to {db}")

    if digest_out is None:
        return
    digest = digest_export.to_digest(hierarchy, taxa, edges, str(source))
    digest_out.parent.mkdir(parents=True, exist_ok=True)
    digest_out.write_text(json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8")
    props = digest["properties"]
    # Offerable count is not decoration: a property with no domain/range is never
    # handed to the extractor, so this is the relation vocabulary that can fire.
    offerable = sum(1 for p in props if p["domain"] and p["range"])
    print(
        f"{digest_out}: {len(digest['classes'])} classes, {len(props)} properties "
        f"({offerable} offerable: domain+range after inheritance), "
        f"{digest_out.stat().st_size / 1024:.0f} KB"
    )


def main() -> int:
    """Project ``--source`` RDF into namespace ``--ns``, reporting what came out."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="RDF file or folder")
    p.add_argument("--ns", required=True, help="graphknows namespace (database mem_<ns>)")
    p.add_argument("--digest-out", type=Path, help="also write a standard ontology digest here")
    p.add_argument(
        "--reset", action="store_true", help="drop the namespace database before writing"
    )
    a = p.parse_args()
    if not a.source.exists():
        raise SystemExit(f"source not found: {a.source}")
    asyncio.run(_run(a.source, a.ns, a.digest_out, a.reset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
