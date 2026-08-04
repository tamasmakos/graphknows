#!/usr/bin/env python
r"""Parse RDF ontologies into the pre-parsed digest graphknows ships and loads.

Why a digest rather than the raw ``.ttl``:

* **No rdflib at runtime.** ``rdflib`` is in the ``assisted`` extra, but the
  ontology layer is always on, including in the default ``llm_free`` mode that a
  bare ``pip install graphknows`` gives you. Reading a plain JSON digest keeps
  the base install free of an RDF stack it would otherwise need on every ingest,
  and makes both modes load the ontology through one identical code path.
* **Startup cost.** Parsing 3 MB of Turtle across 12 modules costs seconds;
  reading the digest costs milliseconds.

The bundled ontology is the Common Core Ontologies: the 11 CCO modules plus the
Familial Relations extension. Two CCO properties matter for how this parses:

* **Opaque IRIs.** CCO names every term ``ont00001774``, so a URI's local name
  is not a label. ``domain``/``range``/``parents`` are therefore resolved
  through a URI->label map built across ALL input files before any term is
  emitted — a per-file parse would leave every cross-module reference as an
  ont-number, which is unreadable to an embedder and useless to the relation
  conformance check.
* **Definitions live in ``skos:definition``.** ``rdfs:comment`` in CCO is an
  editorial note ("This class was added at the request of..."), so it is only a
  fallback. ``skos:scopeNote`` carries usage guidance written in the same
  register as conversational prose and is appended to the definition: it is
  extra retrieval surface for the same term, and it is what makes a class
  reachable from text that never says its label.

Only OBJECT properties are kept. A ``owl:DatatypeProperty`` describes a literal
attribute, not a link between two entities, so it can never be an
ENTITY->ENTITY relation and would only dilute the relation label space handed to
the extractor.

Usage::

    python scripts/build_ontology_digest.py \\
        --source graphknows/ontology/assets/cco-modules \\
        --source graphknows/ontology/assets/cco-extensions/FamilialRelationsOntology.ttl \\
        --out graphknows/ontology/assets/cco/cco.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Extensions deliberately excluded from the bundle, recorded here because "why
# is Barcode missing" is otherwise unanswerable: ModalRelationOntology is 278
# modal-logic properties with ZERO definitions, BarcodeOntology is 44 barcode
# symbologies, and neither describes anything a conversation asserts.

# CCO's "is curated in ontology" annotation — every term carries the module IRI
# it belongs to. Read for the ``module`` field, and for _DEFAULT_DOMAIN_RANGE.
_CURATED_IN = "https://www.commoncoreontologies.org/ont00001760"

# Domain/range asserted per module, for modules that declare none.
#
# This is the one place the digest states something its source does not, so it
# is worth being precise about why. Relation-property selection is GATED on
# domain and range (``stm/ingest.py::_relation_labels``): a property declaring
# neither is never offered to the extractor. That gate is load-bearing — ranking
# properties by whole-chunk cosine instead measured ZERO relations over 60 real
# turns — so it is not something to relax.
#
# FamilialRelationsOntology declares no domain or range on any of its 86
# properties, and none is inheritable: the whole chain up through "is sibling
# of" to "has familial relationship to" is empty. Left as-is, every familial
# relation is permanently unofferable — which is precisely the conversational
# vocabulary this bundle is for. Person->Person is true of the module by
# construction (it is the ontology of relations between family members), so
# asserting it here restores the module without weakening the gate.
#
# Other modules' undeclared properties (TimeOntology's "has inside instant",
# ExtendedRelationOntology's "has object") are left unofferable on purpose: they
# are generic BFO plumbing with no single defensible domain.
_DEFAULT_DOMAIN_RANGE: dict[str, tuple[str, str]] = {
    "FamilialRelationsOntology": ("Person", "Person"),
}


def _files(sources: list[Path]) -> list[Path]:
    """Every ``.ttl`` under the given files/folders, sorted for a stable order.

    Term order fixes the embedding matrix's row order downstream, and a cached
    matrix is only meaningful against the term list that built it.
    """
    out: list[Path] = []
    for src in sources:
        if src.is_dir():
            out.extend(sorted(src.rglob("*.ttl")))
        elif src.is_file():
            out.append(src)
        else:
            raise SystemExit(f"source not found: {src}")
    return out


def build(sources: list[Path]) -> dict[str, Any]:
    """Parse *sources* into ``{"classes": [...], "properties": [...]}``."""
    import rdflib  # build-time only; never imported at runtime
    from rdflib import OWL, RDF, RDFS
    from rdflib.namespace import SKOS

    graph = rdflib.Graph()
    paths = _files(sources)
    for path in paths:
        graph.parse(str(path), format="turtle")

    # URI -> label across every input, so a cross-module domain/range resolves.
    labels: dict[str, str] = {
        str(s): str(o) for s, o in graph.subject_objects(RDFS.label) if str(o).strip()
    }

    def label(subject: Any) -> str:
        return labels.get(str(subject), "")

    def definition(subject: Any) -> str:
        text = graph.value(subject, SKOS.definition) or graph.value(subject, RDFS.comment) or ""
        note = graph.value(subject, SKOS.scopeNote) or ""
        # URLs are embedded verbatim otherwise, and several scopeNotes are bare
        # pointers ("please refer to https://github.com/..."), which is noise in
        # the vector the whole match is decided on.
        return " ".join(re.sub(r"https?://\S+", "", f"{text} {note}").split())

    def refs(subject: Any, predicate: Any) -> list[str]:
        """Labels of ``subject predicate ?o``, skipping blank-node restrictions.

        An anonymous ``owl:Restriction``/union is not an addressable term, and a
        term outside these inputs has no label to resolve to; both are dropped
        rather than emitted as an ont-number.
        """
        return sorted({lbl for o in graph.objects(subject, predicate) if (lbl := label(o))})

    curated_in = rdflib.URIRef(_CURATED_IN)

    def module(subject: Any) -> str:
        """The CCO module this term is curated in, by its own annotation."""
        value = graph.value(subject, curated_in)
        return str(value).rstrip("/").rsplit("/", 1)[-1] if value else ""

    classes = [
        {
            "uri": str(s),
            "label": lbl,
            "definition": definition(s),
            "parents": refs(s, RDFS.subClassOf),
            "module": module(s),
        }
        for s in sorted(graph.subjects(RDF.type, OWL.Class), key=str)
        if (lbl := label(s))
    ]

    properties = []
    for s in sorted(graph.subjects(RDF.type, OWL.ObjectProperty), key=str):
        if not (lbl := label(s)):
            continue
        mod = module(s)
        domain, rng = refs(s, RDFS.domain), refs(s, RDFS.range)
        if not domain and not rng and (fallback := _DEFAULT_DOMAIN_RANGE.get(mod)):
            domain, rng = [fallback[0]], [fallback[1]]
        properties.append(
            {
                "uri": str(s),
                "label": lbl,
                "definition": definition(s),
                "domain": domain,
                "range": rng,
                "parents": refs(s, RDFS.subPropertyOf),
                "module": mod,
            }
        )

    return {
        "source": ", ".join(p.name for p in paths),
        "classes": classes,
        "properties": properties,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="RDF file or folder (repeatable)",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    digest = build(a.source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8")
    classes, props = digest["classes"], digest["properties"]
    undefined = sum(1 for t in (*classes, *props) if not t["definition"])
    # Offerable count is not decoration: a property with no domain/range is
    # never handed to the extractor, so this is the size of the relation
    # vocabulary that can actually fire.
    offerable = sum(1 for t in props if t["domain"] and t["range"])
    print(
        f"{a.out}: {len(classes)} classes, {len(props)} object properties "
        f"({offerable} offerable: domain+range declared), "
        f"{undefined} without a definition, {a.out.stat().st_size / 1024:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
