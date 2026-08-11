#!/usr/bin/env python
r"""Parse RDF ontologies into the pre-parsed digest graphknows ships and loads.

This is how you bring your own ontology. graphknows bundles a CCO digest as the
default, but any RDF source becomes usable by digesting it here and pointing
``GRAPHKNOWS_ONTOLOGY`` at the result — the loader reads a digest identically
whichever ontology produced it. It lives in the package rather than in the repo's
``scripts/`` for exactly that reason: it is a user-facing tool, not a build step
of ours. Needs the ``assisted`` extra (rdflib); nothing else here does.

Why a digest rather than the raw ``.ttl``:

* **No rdflib at runtime.** ``rdflib`` is in the ``assisted`` extra, but the
  ontology layer is always on, including in the default ``llm_free`` mode that a
  bare ``pip install graphknows`` gives you. Reading a plain JSON digest keeps
  the base install free of an RDF stack it would otherwise need on every ingest,
  and makes both modes load the ontology through one identical code path.
* **Startup cost.** Parsing 3 MB of Turtle across 12 modules costs seconds;
  reading the digest costs milliseconds.

The bundled ontology is the Common Core Ontologies: the 11 CCO modules, the
Familial Relations extension, and **BFO**, which is not optional. Four things
about CCO decide how this parses:

* **Opaque IRIs.** CCO names every term ``ont00001774``, so a URI's local name
  is not a label. ``domain``/``range``/``parents`` are therefore resolved
  through a URI->label map built across ALL input files before any term is
  emitted — a per-file parse would leave every cross-module reference as an
  ont-number, which is unreadable to an embedder and useless to the relation
  conformance check.
* **CCO types its relations against BFO.** ``rdfs:domain obo:BFO_0000015``
  (process), ``obo:BFO_0000004`` (independent continuant). BFO is an *import*
  — CCO's own ``catalog-v001.xml`` resolves it to ``../cco-imports/bfo-core.ttl``
  — so a map built from the CCO modules alone resolves every one of those
  references to nothing and DROPS the constraint. That single omission made
  199 of 224 properties look untyped, and since property selection is gated on
  domain and range, it silently emptied the relation vocabulary. Parsing BFO
  alongside takes typed properties from 25 to 173. BFO's own classes are
  emitted too: they are what the ``parents`` chain terminates in, and without
  them subsumption stops at the CCO boundary instead of reaching
  ``material entity`` / ``independent continuant``.
* **Domain and range hide inside class expressions.** A range is frequently an
  anonymous ``owl:unionOf`` / ``owl:intersectionOf`` node rather than a named
  class. Those are unwrapped to the set of named classes they mention; an
  ``owl:complementOf`` ("not a spatial region") carries no positive label and is
  skipped rather than inverted.
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

Usage — your own ontology::

    python -m graphknows.ontology.digest --source my-ontology.ttl --out my.json
    export GRAPHKNOWS_ONTOLOGY=my.json

Usage — regenerating the bundled CCO digest::

    python -m graphknows.ontology.digest \\
        --source graphknows/ontology/assets/cco-modules \\
        --source graphknows/ontology/assets/cco-extensions/FamilialRelationsOntology.ttl \\
        --source graphknows/ontology/assets/cco-imports/bfo-core.ttl \\
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


class _Reader:
    """Term-level readers over one parsed graph.

    A class rather than nested closures so each rule is separately readable:
    every method below encodes one thing CCO does that a naive parse gets wrong.
    """

    def __init__(self, graph: Any, labels: dict[str, str]) -> None:
        import rdflib

        self.g = graph
        self.labels = labels
        self.curated_in = rdflib.URIRef(_CURATED_IN)

    def label(self, subject: Any) -> str:
        return self.labels.get(str(subject), "")

    def definition(self, subject: Any) -> str:
        """``skos:definition``, falling back to ``rdfs:comment``, plus the scope note."""
        from rdflib import RDFS
        from rdflib.namespace import SKOS

        text = self.g.value(subject, SKOS.definition) or self.g.value(subject, RDFS.comment) or ""
        note = self.g.value(subject, SKOS.scopeNote) or ""
        # URLs are embedded verbatim otherwise, and several scopeNotes are bare
        # pointers ("please refer to https://github.com/..."), which is noise in
        # the vector the whole match is decided on.
        return " ".join(re.sub(r"https?://\S+", "", f"{text} {note}").split())

    def named(self, node: Any, depth: int = 0) -> set[str]:
        """Class labels ``node`` denotes, unwrapping anonymous class expressions.

        A blank node here is an ``owl:unionOf`` / ``owl:intersectionOf`` /
        ``owl:Restriction``, not an addressable term. ``unionOf`` means "any of
        these", which is exactly a set of acceptable labels. ``intersectionOf``
        is strictly narrower than the set, but reading it as the set is the
        permissive direction and this gate enriches rather than filters.
        ``complementOf`` carries no positive label and is skipped rather than
        inverted. Depth-bounded: BFO nests these a few levels.
        """
        import rdflib
        from rdflib import OWL

        if depth > 4:
            return set()
        # isinstance, NOT a "_:" prefix test: rdflib's BNode stringifies to a bare
        # id ("ne038ab1...") with no "_:" on it, so the prefix test this used to
        # run was False for every blank node and the whole unwrapping branch below
        # was unreachable. Every union/intersection domain and range silently read
        # as empty.
        if not isinstance(node, rdflib.BNode):
            return {lbl} if (lbl := self.label(node)) else set()
        out: set[str] = set()
        for pred in (OWL.unionOf, OWL.intersectionOf):
            for collection in self.g.objects(node, pred):
                for member in rdflib.collection.Collection(self.g, collection):
                    out |= self.named(member, depth + 1)
        return out

    def genus_parents(self, subject: Any) -> list[str]:
        """Parents a DEFINED class states inside ``owl:equivalentClass``.

        A defined class asserts no ``rdfs:subClassOf`` at all: its genus sits in
        ``owl:equivalentClass [ owl:intersectionOf ( <genus> <restriction>... ) ]``.
        Reading only the asserted triple leaves ~447 CCO classes parentless, and
        a parentless class truncates every ancestor walk built on ``parents`` —
        the relation-offer widening in ``_relation_labels`` and the BROADER hop
        the ontology channel takes both stop short at exactly those terms.

        Only ``intersectionOf`` members count, and only through
        :meth:`named`, which returns nothing for a restriction — "assists some
        Person" says who a guide dog helps, not what it is. ``unionOf`` members
        are the opposite direction: ``Pet == Dog or Cat`` means Pet SUBSUMES
        them, so reading them here would put the general term under the specific
        ones. A NAMED equivalent class is a synonym, not a parent, and is
        likewise skipped.
        """
        import rdflib
        from rdflib import OWL

        out: set[str] = set()
        for equiv in self.g.objects(subject, OWL.equivalentClass):
            if not isinstance(equiv, rdflib.BNode):
                continue
            for collection in self.g.objects(equiv, OWL.intersectionOf):
                for member in rdflib.collection.Collection(self.g, collection):
                    out |= self.named(member)
        return sorted(out)

    def refs(self, subject: Any, predicate: Any) -> list[str]:
        """Labels of ``subject predicate ?o``, through class expressions."""
        out: set[str] = set()
        for obj in self.g.objects(subject, predicate):
            out |= self.named(obj)
        return sorted(out)

    def constraint(
        self, prop: Any, predicate: Any, seen: frozenset[str] = frozenset()
    ) -> list[str]:
        """Domain/range for ``prop``, inherited from superproperties when absent.

        CCO declares a constraint once on the general property and leaves the
        specialisations to inherit it: ``has process part`` states no domain of
        its own and takes BFO's. Reading only the direct triple drops those.
        """
        from rdflib import RDFS

        if str(prop) in seen:
            return []
        if direct := self.refs(prop, predicate):
            return direct
        inherited: set[str] = set()
        for parent in self.g.objects(prop, RDFS.subPropertyOf):
            inherited.update(self.constraint(parent, predicate, seen | {str(prop)}))
        return sorted(inherited)

    def module(self, subject: Any) -> str:
        """The CCO module this term is curated in, by its own annotation."""
        value = self.g.value(subject, self.curated_in)
        return str(value).rstrip("/").rsplit("/", 1)[-1] if value else ""


def build(sources: list[Path]) -> dict[str, Any]:
    """Parse *sources* into ``{"classes": [...], "properties": [...]}``."""
    # Deferred, and the only door into rdflib in this module: `_Reader` is built
    # here and nowhere else. rdflib ships in the `assisted` extra, so a base
    # install can import this module (and load a digest) without an RDF stack.
    try:
        import rdflib
        from rdflib import OWL, RDF, RDFS
    except ImportError as exc:
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("Building an ontology digest", "assisted") from exc

    graph = rdflib.Graph()
    paths = _files(sources)
    for path in paths:
        graph.parse(str(path), format="turtle")

    # URI -> label across every input, so a cross-module domain/range resolves.
    # BFO is one of those inputs, and CCO types its relations against it.
    labels: dict[str, str] = {
        str(s): str(o) for s, o in graph.subject_objects(RDFS.label) if str(o).strip()
    }
    r = _Reader(graph, labels)

    classes = [
        {
            "uri": str(s),
            "label": lbl,
            "definition": r.definition(s),
            "parents": sorted(set(r.refs(s, RDFS.subClassOf)) | set(r.genus_parents(s))),
            "module": r.module(s),
        }
        for s in sorted(graph.subjects(RDF.type, OWL.Class), key=str)
        if (lbl := r.label(s))
    ]

    properties = []
    for s in sorted(graph.subjects(RDF.type, OWL.ObjectProperty), key=str):
        if not (lbl := r.label(s)):
            continue
        mod = r.module(s)
        domain, rng = r.constraint(s, RDFS.domain), r.constraint(s, RDFS.range)
        if not domain and not rng and (fallback := _DEFAULT_DOMAIN_RANGE.get(mod)):
            domain, rng = [fallback[0]], [fallback[1]]
        properties.append(
            {
                "uri": str(s),
                "label": lbl,
                "definition": r.definition(s),
                "domain": domain,
                "range": rng,
                "parents": r.refs(s, RDFS.subPropertyOf),
                "module": mod,
            }
        )

    return {
        "source": ", ".join(p.name for p in paths),
        "classes": classes,
        "properties": properties,
    }


def main() -> int:
    """Digest ``--source`` RDF into ``--out`` JSON, reporting what came out."""
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
