#!/usr/bin/env python
r"""Parse an RDF ontology into the pre-parsed digest graphknows ships and loads.

Why a digest rather than the raw ``.ttl``:

* **No rdflib at runtime.** ``rdflib`` is in the ``assisted`` extra, but the
  ontology layer is now always on, including in the default ``llm_free`` mode
  that a bare ``pip install graphknows`` gives you. Reading a plain JSON digest
  keeps the base install free of an RDF stack it would otherwise need on every
  ingest.
* **schema.org does not speak OWL.** It declares ``rdfs:Class`` (not
  ``owl:Class``) and ``schema:domainIncludes`` / ``schema:rangeIncludes`` (not
  ``rdfs:domain`` / ``rdfs:range``), and both are MULTI-valued. Normalising that
  here means the runtime loader has one shape to understand instead of a
  per-vocabulary parser.
* **Startup cost.** Parsing 1.1 MB of Turtle costs seconds; reading the digest
  costs milliseconds.

This script produces the ENTITY-TYPING vocabulary. The digest it emits from
schema.org is what types entities and steers entity labels. The RELATION
vocabulary is a different file with the same shape —
``assets/profile/personal-profile.json``, 9 classes and 25 properties — and it is
hand-written, not generated here: it is assembled from three namespaces
(schema.org, the Relationship vocabulary, plus 3 invented terms), so there is no
single RDF source to parse, and the curation IS the deliverable. Its shape is
this one, so the loader and every consumer are unchanged.

Only OBJECT properties are kept. A property whose range is entirely datatypes
(``Text``, ``Number``, ``Date``, …) describes a literal attribute, not a link
between two entities, so it can never be an ENTITY->ENTITY relation and would
only dilute the relation label space handed to the extractor.

Usage::

    python scripts/build_ontology_digest.py \\
        --url https://schema.org/version/latest/schemaorg-current-https.ttl \\
        --out graphknows/ontology/assets/schemaorg/schemaorg.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

# schema.org range targets that are literals rather than entities.
DATATYPES = frozenset(
    {
        "Text",
        "Number",
        "Date",
        "DateTime",
        "Boolean",
        "Time",
        "URL",
        "Integer",
        "Float",
        "XPathType",
        "CssSelectorType",
        "PronounceableText",
        "DataType",
    }
)


def _local(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def build(source: Path, namespace: str = "https://schema.org/") -> dict[str, Any]:
    """Parse *source* into ``{"classes": [...], "properties": [...]}``.

    Terms are restricted to *namespace*. schema.org's Turtle carries
    ``owl:equivalentClass`` links to other vocabularies, and those targets are
    themselves typed ``rdfs:Class`` in the same file — so a type-only filter
    swept in 77 foreign classes: SNOMED codes (``116154003``), LRMoo
    (``F31_Performance``), DCMI, and an FIBO ``BankAccount``. They are not junk
    in principle, but they are unusable as extraction labels (a bare clinical
    code cannot be embedded meaningfully) and they were minting ENTITY types
    like ``BANKACCOUNT`` on dialogue.
    """
    import rdflib  # build-time only; never imported at runtime
    from rdflib import RDF, RDFS

    graph = rdflib.Graph()
    graph.parse(str(source), format="turtle")
    sdo = rdflib.Namespace("https://schema.org/")

    def label(subject: Any) -> str:
        value = graph.value(subject, RDFS.label)
        return str(value) if value else _local(str(subject))

    def definition(subject: Any) -> str:
        value = graph.value(subject, RDFS.comment)
        # Definitions carry HTML in schema.org ("<a href=...>"). The embedder
        # sees this text verbatim, so strip the markup rather than embed tags.
        return " ".join(str(value).split()) if value else ""

    classes = [
        {"uri": str(s), "label": label(s), "definition": definition(s)}
        for s in sorted(graph.subjects(RDF.type, RDFS.Class), key=str)
        if str(s).startswith(namespace)
    ]

    properties: list[dict[str, Any]] = []
    for s in sorted(graph.subjects(RDF.type, RDF.Property), key=str):
        if not str(s).startswith(namespace):
            continue
        ranges = [_local(str(r)) for r in graph.objects(s, sdo.rangeIncludes)]
        if not any(r not in DATATYPES for r in ranges):
            continue  # datatype-only property: an attribute, not a relation
        properties.append(
            {
                "uri": str(s),
                "label": label(s),
                "definition": definition(s),
                "domain": sorted({_local(str(d)) for d in graph.objects(s, sdo.domainIncludes)}),
                "range": sorted({r for r in ranges if r not in DATATYPES}),
            }
        )

    return {"source": source.name, "classes": classes, "properties": properties}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", help="download the ontology from here first")
    p.add_argument("--source", type=Path, help="local RDF file (skips --url)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--namespace",
        default="https://schema.org/",
        help="keep only terms whose IRI starts with this (see build()'s docstring)",
    )
    a = p.parse_args()

    source = a.source
    if source is None:
        if not a.url:
            p.error("one of --url or --source is required")
        source = Path("/tmp/_ontology_download.ttl")
        print(f"downloading {a.url}", file=sys.stderr)
        urllib.request.urlopen(a.url)  # operator-supplied build-time URL
        with urllib.request.urlopen(a.url) as fh, source.open("wb") as out:
            out.write(fh.read())

    digest = build(source, a.namespace)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8")
    size_kb = a.out.stat().st_size / 1024
    print(
        f"{a.out}: {len(digest['classes'])} classes, "
        f"{len(digest['properties'])} object properties, {size_kb:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
