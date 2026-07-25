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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("graphknows.ontology.loader")

# Suffix -> rdflib format. ".owl" is RDF/XML but rdflib does not infer it.
_RDF_FORMATS: dict[str, str] = {
    ".ttl": "turtle",
    ".owl": "xml",
    ".rdf": "xml",
    ".xml": "xml",
    ".jsonld": "json-ld",
    ".nt": "nt",
    ".n3": "n3",
}


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
    domain: str = ""
    range: str = ""
    parents: tuple[str, ...] = ()
    source: str = ""

    @property
    def index_text(self) -> str:
        """The string that represents this term in the embedding catalog."""
        return f"{self.label}. {self.definition}".strip().rstrip(".")


def _rdf_label(graph: object, node: object) -> str:
    from rdflib import RDFS  # type: ignore

    label = graph.value(node, RDFS.label)  # type: ignore[attr-defined]
    if label:
        return str(label)
    text = str(node)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    return text


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


def _collect_terms(graph: Any, subjects: Any, kind: str, source: str) -> list[OntologyTerm]:
    """Build :class:`OntologyTerm` records for one kind of subject."""
    from rdflib import RDFS  # type: ignore
    from rdflib.namespace import DCTERMS, SKOS  # type: ignore

    terms: list[OntologyTerm] = []
    for node in subjects:
        # Blank nodes are anonymous restrictions/unions, not addressable terms.
        if not str(node).startswith(("http://", "https://", "urn:")):
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
                domain=_local_name(graph.value(node, RDFS.domain))
                if graph.value(node, RDFS.domain)
                else "",
                range=_local_name(graph.value(node, RDFS.range))
                if graph.value(node, RDFS.range)
                else "",
                parents=tuple(
                    sorted(
                        _local_name(p)
                        for p in graph.objects(
                            node, RDFS.subClassOf if kind == "class" else RDFS.subPropertyOf
                        )
                    )
                ),
                source=source,
            )
        )
    return terms


def _terms_from_rdf(path: Path) -> list[OntologyTerm]:
    """Parse one RDF/OWL/TTL/JSON-LD file into terms. Empty if it will not parse."""
    import rdflib  # type: ignore
    from rdflib import OWL, RDF, RDFS  # type: ignore

    graph = rdflib.Graph()
    fmt = _RDF_FORMATS.get(path.suffix.lower())
    try:
        graph.parse(str(path), format=fmt) if fmt else graph.parse(str(path))
    except Exception as exc:
        log.debug("Not parseable as RDF (%s): %s", path.name, exc)
        return []

    class_nodes = set(graph.subjects(RDF.type, OWL.Class)) | set(
        graph.subjects(RDF.type, RDFS.Class)
    )
    property_nodes = (
        set(graph.subjects(RDF.type, OWL.ObjectProperty))
        | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        | set(graph.subjects(RDF.type, RDF.Property))
    )
    source = str(path)
    # Sorted, not set-ordered: term order determines the embedding matrix's row
    # order, and rdflib subject sets iterate differently between processes
    # (randomized string hashing). An unstable order silently pairs a cached
    # matrix with a different term list.
    return [
        *_collect_terms(graph, sorted(class_nodes, key=str), "class", source),
        *_collect_terms(graph, sorted(property_nodes, key=str), "property", source),
    ]


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
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in _RDF_FORMATS and p.is_file())


def load_ontology_terms(source: str | Path) -> list[OntologyTerm]:
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
        Class and property terms, ordered classes-then-properties by file.

    Raises:
        FileNotFoundError: When ``source`` does not exist.
        ValueError: When nothing in ``source`` yielded a single term.
    """
    resolved = Path(source)
    if not resolved.exists():
        raise FileNotFoundError(f"Ontology source not found: {resolved}")

    merged: dict[tuple[str, str], OntologyTerm] = {}
    for path in _ontology_files(resolved):
        # A summary digest is only consulted when the file is not real RDF.
        for term in _terms_from_rdf(path) or _terms_from_summary_json(path):
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
    return list(merged.values())


def load_rdf_ontology_file(path: str | Path) -> RdfOntologyManifest:
    """Load labels from one explicit RDF/OWL ontology file.

    This function is intentionally strict for LLM-assisted DSPy extraction:
    callers must provide an RDF/OWL file, and missing labels are errors rather
    than reasons to fall back to taxonomy YAML or generated defaults.
    """
    import rdflib  # type: ignore
    from rdflib import OWL, RDF, RDFS  # type: ignore

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"RDF ontology file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"RDF ontology path is not a file: {resolved}")

    graph = rdflib.Graph()
    graph.parse(str(resolved))

    class_nodes = set(graph.subjects(RDF.type, OWL.Class)) | set(
        graph.subjects(RDF.type, RDFS.Class)
    )
    property_nodes = (
        set(graph.subjects(RDF.type, OWL.ObjectProperty))
        | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        | set(graph.subjects(RDF.type, RDF.Property))
    )

    classes = sorted({_rdf_label(graph, node) for node in class_nodes if _rdf_label(graph, node)})
    properties = sorted(
        {_rdf_label(graph, node) for node in property_nodes if _rdf_label(graph, node)}
    )

    if not classes:
        raise ValueError(f"RDF ontology file has no owl:Class/rdfs:Class labels: {resolved}")
    if not properties:
        raise ValueError(f"RDF ontology file has no RDF/OWL property labels: {resolved}")

    context = {
        "file": str(resolved),
        "classes": classes,
        "properties": properties,
    }
    return RdfOntologyManifest(
        file_path=str(resolved),
        class_labels=classes,
        property_labels=properties,
        context=json.dumps(context, ensure_ascii=False),
    )
