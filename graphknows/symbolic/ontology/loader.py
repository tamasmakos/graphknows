"""Ontology loader — parses ontology files into terms and label lists.

Two views coexist:

- :func:`load_rdf_ontology_file` returns *label lists* for LLM prompt injection
  (the llm_assisted path), strictly, from one explicit RDF/OWL file.
- :func:`load_ontology_terms` returns :class:`OntologyTerm` records carrying the
  natural-language **definition** of every class and property. Those definitions
  are what the embedding catalog matches chunk text against, so both memory
  modes can steer extraction with an ontology.

Both take a path the caller supplies. Bundled ontologies under ``assets/`` are
addressed the same way as any other file or folder.

Three input shapes are recognised, in this order: our pre-built **digest** JSON
(``assets/cco/cco.json``, the always-on default), then RDF/OWL/TTL,
then the legacy ``*-summary.jsonld`` label lists. The digest is first because it
is the only shape parsed with the standard library alone — ``rdflib`` lives in
the ``assisted`` extra, and the default ``llm_free`` install does not have it.

Every RDF document is opened through
:func:`~graphknows.symbolic.ontology.rdf_io.parse_file`, which is what keeps this
loader offline: it takes a ``Path``, opens it, and hands rdflib a file object and
an explicit format, so no source string ever reaches a parser that would resolve
it over the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("graphknows.symbolic.ontology.loader")


@dataclass(frozen=True)
class RdfOntologyManifest:
    """Strict RDF/OWL ontology view for DSPy relation extraction."""

    file_path: str
    class_labels: list[str]
    property_labels: list[str]
    context: str


@dataclass(frozen=True)
class OntologyTerm:
    """One ontology class or property, with the text that describes it.

    The ``definition`` is the load-bearing field: it is what gets embedded and
    cosine-matched against chunk text, so a term with a good definition is
    reachable from prose that never says its label. Terms with no definition
    still index on their label alone.
    """

    uri: str
    label: str
    definition: str
    kind: str  # "class" | "property"
    # Multi-valued because a vocabulary may declare several: schema.org's
    # ``about`` declared six domains. Conformance is any-match — a relation
    # conforms when its subject satisfies ANY declared domain (see
    # ``_domain_admits``). Tuples, not lists: this dataclass is frozen and hashed.
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    # ``skos:broader`` parents, which SKOS does not define as transitive: a
    # consumer may report such a parent but must not climb above it. ``parents``
    # (``rdfs:subClassOf`` / ``rdfs:subPropertyOf``) is transitive and composes.
    parents_nontransitive: tuple[str, ...] = ()
    source: str = ""
    # The sub-ontology a term is curated in ("AgentOntology"), when the source
    # says so. Carried onto the ONTOLOGY_CLASS node so a match can be traced to
    # the module that supplied it; never used for selection.
    module: str = ""
    # An owl:FunctionalProperty has at most one value per subject, which is what
    # lets an invalidation pass tombstone the older assertion instead of keeping
    # both. Always False for classes.
    functional: bool = False
    # skos:altLabel — synonyms a mention may use instead of the preferred label
    # ("purchase" for "Buying"). Sorted at harvest time so row order (and the
    # digest) stay stable across processes.
    alt_labels: tuple[str, ...] = ()
    # skos:example — usage text embedded straight from the source vocabulary,
    # often the richest signal a term has when its definition is terse.
    examples: tuple[str, ...] = ()

    @property
    def index_text(self) -> str:
        """The string that represents this term in the embedding catalog.

        Alt labels and examples are appended after the definition: they widen
        what a chunk can match on without displacing the definition, which
        stays first and therefore dominates the embedding least-lossily.
        """
        parts = [self.label, self.definition, *self.alt_labels, *self.examples]
        return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())


def _text_value(graph: Any, node: Any, *predicates: Any) -> str:
    """First non-empty literal among ``predicates`` for ``node``."""
    for pred in predicates:
        value = graph.value(node, pred)
        if value:
            return str(value).strip()
    return ""


def _local_name(node: Any) -> str:
    """Human-readable tail of a URI, for domain/range/parent references."""
    text = str(node)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    return text


def _collect_terms(
    graph: Any,
    subjects: Any,
    kind: str,
    source: str,
    functional_nodes: frozenset[Any] = frozenset(),
) -> tuple[list[OntologyTerm], int]:
    """Build :class:`OntologyTerm` records for one kind of subject.

    ``functional_nodes`` is only ever non-empty for properties; classes never
    match it, so they always come out with ``functional=False``.

    Returns the terms plus a count of blank-node subjects skipped (anonymous
    restrictions/unions) — an import tally the caller aggregates, since those
    subjects never become an addressable term.
    """
    from rdflib import RDFS  # type: ignore
    from rdflib.namespace import DCTERMS, SKOS  # type: ignore

    terms: list[OntologyTerm] = []
    dropped = 0
    for node in subjects:
        # Blank nodes are anonymous restrictions/unions, not addressable terms.
        if not str(node).startswith(("http://", "https://", "urn:")):
            dropped += 1
            continue
        label = _text_value(graph, node, SKOS.prefLabel, RDFS.label) or _local_name(node)
        if not label:
            continue
        terms.append(
            OntologyTerm(
                uri=str(node),
                label=label,
                # skos:definition first: it is written for humans, whereas
                # rdfs:comment is often an editorial note.
                definition=_text_value(
                    graph, node, SKOS.definition, RDFS.comment, DCTERMS.description
                ),
                kind=kind,
                domain=tuple(sorted(_local_name(d) for d in graph.objects(node, RDFS.domain))),
                range=tuple(sorted(_local_name(r) for r in graph.objects(node, RDFS.range))),
                parents=tuple(
                    sorted(
                        _local_name(p)
                        for p in graph.objects(
                            node, RDFS.subClassOf if kind == "class" else RDFS.subPropertyOf
                        )
                    )
                ),
                parents_nontransitive=tuple(
                    sorted(_local_name(p) for p in graph.objects(node, SKOS.broader))
                ),
                alt_labels=tuple(
                    sorted(str(a).strip() for a in graph.objects(node, SKOS.altLabel) if a)
                ),
                examples=tuple(
                    sorted(str(e).strip() for e in graph.objects(node, SKOS.example) if e)
                ),
                source=source,
                functional=node in functional_nodes,
            )
        )
    return terms, dropped


def _rdf_terms(graph: Any, source: str) -> tuple[list[OntologyTerm], int]:
    """Every class and property term in an already-parsed graph, plus dropped-node count."""
    from rdflib import OWL, RDF, RDFS  # type: ignore

    class_nodes = set(graph.subjects(RDF.type, OWL.Class)) | set(
        graph.subjects(RDF.type, RDFS.Class)
    )
    property_nodes = (
        set(graph.subjects(RDF.type, OWL.ObjectProperty))
        | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        | set(graph.subjects(RDF.type, RDF.Property))
    )
    functional_nodes = frozenset(graph.subjects(RDF.type, OWL.FunctionalProperty))
    # Sorted, not set-ordered: term order determines the embedding matrix's row
    # order, and rdflib subject sets iterate differently between processes
    # (randomized string hashing). An unstable order silently pairs a cached
    # matrix with a different term list.
    class_terms, class_dropped = _collect_terms(
        graph, sorted(class_nodes, key=str), "class", source
    )
    prop_terms, prop_dropped = _collect_terms(
        graph, sorted(property_nodes, key=str), "property", source, functional_nodes
    )
    return [*class_terms, *prop_terms], class_dropped + prop_dropped


def _terms_from_rdf(path: Path) -> tuple[list[OntologyTerm], int]:
    """Parse one local RDF file into terms. Empty when it will not parse.

    Permissive about CONTENT — this is the folder-walking path, where a source
    directory legitimately mixes ontologies with digests and junk, and one bad
    file must not cost the other forty.

    Not permissive about FORMAT. A suffix rdflib has no parser for reaches here
    only when the caller named that file explicitly (the folder walk filters on
    the same table), and answering "no terms found" would hide the one fact that
    fixes it: it needs converting first.
    """
    from graphknows.symbolic.ontology import rdf_io

    try:
        graph = rdf_io.parse_file(path)
    except ValueError:
        raise
    except Exception as exc:
        log.debug("Not parseable as RDF (%s): %s", path.name, exc)
        return [], 0
    return _rdf_terms(graph, str(path))


def _digest_payload(path: Path) -> dict[str, Any] | None:
    """The parsed digest, or ``None`` when ``path`` is not one.

    A digest is our own pre-parsed shape (``python -m graphknows.symbolic.ontology.digest``):
    ``{"classes": [{"uri", "label", "definition"}], "properties": [...]}``. It is
    recognised BEFORE any RDF path so the bundled CCO ontology — the
    default in every mode — never needs ``rdflib``, which ships only in the
    ``assisted`` extra. rdflib would in fact "parse" this file as JSON-LD and
    return nothing, so the ordering is what makes the default path work at all.
    """
    if path.suffix.lower() not in (".json", ".jsonld"):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("Not readable as JSON (%s): %s", path.name, exc)
        return None
    if not isinstance(data, dict):
        return None
    entries = [*(data.get("classes") or []), *(data.get("properties") or [])]
    # The sibling ``*-summary.jsonld`` files carry plain label STRINGS here, not
    # records; those belong to _terms_from_summary_json.
    if not entries or not all(isinstance(e, dict) and e.get("label") for e in entries):
        return None
    return data


def _terms_from_digest(path: Path) -> tuple[list[OntologyTerm], int]:
    """Parse a pre-built ontology digest. Pure ``json`` — never imports rdflib.

    Returns the terms plus the digest's own ``dropped_expressions`` count
    (anonymous class/property expressions the digest builder saw but could
    not name), so a digest-sourced ``SOURCE`` row reports the real tally
    instead of the RDF-path-only 0 every digest import used to report.
    """
    data = _digest_payload(path)
    if data is None:
        return [], 0
    source = str(path)
    terms = [
        OntologyTerm(
            uri=str(entry["uri"]),
            label=str(entry["label"]),
            definition=str(entry.get("definition") or ""),
            kind=kind,
            domain=tuple(entry.get("domain") or ()),
            range=tuple(entry.get("range") or ()),
            parents=tuple(entry.get("parents") or ()),
            parents_nontransitive=tuple(entry.get("parents_nontransitive") or ()),
            source=source,
            module=str(entry.get("module") or ""),
            functional=bool(entry.get("functional")),
            alt_labels=tuple(entry.get("altLabels") or ()),
            examples=tuple(entry.get("examples") or ()),
        )
        for kind, key in (("class", "classes"), ("property", "properties"))
        for entry in data.get(key) or []
    ]
    return terms, int(data.get("dropped_expressions") or 0)


def _terms_from_summary_json(path: Path) -> list[OntologyTerm]:
    """Parse a ``*-summary.jsonld`` digest (``classes``/``properties`` lists).

    These sit beside the real ontologies in the asset bundle and are not valid
    RDF, but they carry a ``class_definitions`` map worth keeping.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict) or not (data.get("classes") or data.get("properties")):
        return []
    definitions: dict[str, str] = data.get("class_definitions") or {}
    stem = path.stem
    terms: list[OntologyTerm] = []
    for kind, key in (("class", "classes"), ("property", "properties")):
        for label in data.get(key) or []:
            if not isinstance(label, str) or not label:
                continue
            terms.append(
                OntologyTerm(
                    uri=f"urn:graphknows:{stem}:{kind}:{label}",
                    label=label,
                    definition=definitions.get(label, ""),
                    kind=kind,
                    source=str(path),
                )
            )
    return terms


def _ontology_files(source: Path) -> list[Path]:
    """Every candidate ontology file under ``source`` (a file or a folder)."""
    if source.is_file():
        return [source]
    from graphknows.symbolic.ontology import rdf_io

    return sorted(
        p for p in source.rglob("*") if p.suffix.lower() in rdf_io.RDF_FORMATS and p.is_file()
    )


@dataclass(frozen=True)
class LoadedOntology:
    """A load's terms plus the import tallies the SOURCE vertex later records."""

    terms: list[OntologyTerm]
    # Blank-node subjects (anonymous restrictions/unions) seen across every RDF
    # file in the source: addressable in the ontology, but never a term.
    dropped_expressions: int = 0


def load_ontology(source: str | Path) -> LoadedOntology:
    """Load every class and property, with definitions, from a file or folder.

    Unlike :func:`load_rdf_ontology_file` this is *permissive*: a folder may mix
    formats and hold non-ontology files, and unparseable entries are skipped
    rather than raising, because pointing ``GRAPHKNOWS_ONTOLOGY`` at a directory
    of mixed assets is an explicitly supported workflow.

    Terms are de-duplicated by ``(kind, label)`` rather than URI: a folder
    typically mixes an ontology with its JSON-LD digest, and two ontologies may
    each define "Person". Downstream all that matters is the label space handed
    to the extractor, so collapsing them is correct — the entry carrying a
    definition wins.

    Args:
        source: An RDF/OWL/TTL/JSON-LD file, or a folder containing them.

    Returns:
        The merged terms (classes-then-properties by file) plus the count of
        blank-node expressions the RDF path dropped along the way.

    Raises:
        FileNotFoundError: When ``source`` does not exist.
        ValueError: When ``source`` is a URL, or nothing in it yielded a term.
    """
    from graphknows.symbolic.ontology import rdf_io

    resolved = rdf_io.local_path(source)
    if not resolved.exists():
        raise FileNotFoundError(f"Ontology source not found: {resolved}")

    merged: dict[tuple[str, str], OntologyTerm] = {}
    dropped_expressions = 0
    for path in _ontology_files(resolved):
        # Digest first — it is the only path that avoids importing rdflib, and
        # rdflib happily "parses" it as JSON-LD into zero terms. A summary
        # digest is only consulted when the file is not real RDF either.
        digest_terms, digest_dropped = _terms_from_digest(path)
        if digest_terms:
            found = digest_terms
            dropped_expressions += digest_dropped
        else:
            rdf_terms, dropped = _terms_from_rdf(path)
            dropped_expressions += dropped
            found = rdf_terms or _terms_from_summary_json(path)
        for term in found:
            key = (term.kind, term.label.casefold())
            existing = merged.get(key)
            if existing is None or (not existing.definition and term.definition):
                merged[key] = term

    if not merged:
        raise ValueError(f"No ontology classes or properties found in: {resolved}")
    log.info(
        "Loaded %d ontology terms (%d classes, %d properties) from %s",
        len(merged),
        sum(1 for t in merged.values() if t.kind == "class"),
        sum(1 for t in merged.values() if t.kind == "property"),
        resolved,
    )
    return LoadedOntology(terms=list(merged.values()), dropped_expressions=dropped_expressions)


def load_ontology_terms(source: str | Path) -> list[OntologyTerm]:
    """Thin wrapper over :func:`load_ontology` for callers that only want terms."""
    return load_ontology(source).terms


def _manifest_from_terms(path: Path, terms: list[OntologyTerm]) -> RdfOntologyManifest:
    """Label-list view of already-parsed terms (the digest path)."""
    classes = sorted({t.label for t in terms if t.kind == "class"})
    properties = sorted({t.label for t in terms if t.kind == "property"})
    if not classes or not properties:
        raise ValueError(f"Ontology has no classes or no properties: {path}")
    return RdfOntologyManifest(
        file_path=str(path),
        class_labels=classes,
        property_labels=properties,
        context=json.dumps(
            {"file": str(path), "classes": classes, "properties": properties}, ensure_ascii=False
        ),
    )


def load_rdf_ontology_file(path: str | Path) -> RdfOntologyManifest:
    """Load labels from one explicit RDF/OWL ontology file.

    This function is intentionally strict for LLM-assisted DSPy extraction:
    callers must provide an RDF/OWL file, and missing labels are errors rather
    than reasons to fall back to taxonomy YAML or generated defaults.

    A pre-built digest (the bundled CCO default) is accepted here too and
    handled without rdflib: ``settings.ontology_source`` defaults to that
    digest.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When ``path`` is a URL, is not a file, names no RDF syntax,
            or declares no classes or no properties. A parse failure raises too:
            unlike the folder walk, one explicitly named file is never skipped.
    """
    from graphknows.symbolic.ontology import rdf_io

    resolved = rdf_io.local_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"RDF ontology file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"RDF ontology path is not a file: {resolved}")

    terms, _dropped = _terms_from_digest(resolved)
    if terms:
        return _manifest_from_terms(resolved, terms)
    rdf_terms, _dropped = _rdf_terms(rdf_io.parse_file(resolved), str(resolved))
    return _manifest_from_terms(resolved, rdf_terms)
