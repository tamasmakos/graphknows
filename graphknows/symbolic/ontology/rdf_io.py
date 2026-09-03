"""Local-only RDF loading: the offline guard, the ``owl:imports`` closure, stable ids.

Everything here runs at install/digest time. ``rdflib`` is imported INSIDE the
functions and never at module scope — it ships in the ``ontology`` extra, and the
runtime path reads a pre-parsed JSON digest that must stay free of an RDF stack.

Four things this module exists to make structural rather than conventional.

**Offline by design.** ``Graph.parse("http://example.org/o.ttl")`` performs
network I/O, and this system runs in an image with no egress
(``HF_HUB_OFFLINE=1``). A convention ("we always pass local paths") is not a
guard, so :func:`parse_file` takes a ``pathlib.Path``, opens it, and hands rdflib
a file OBJECT plus an explicit ``format=``. rdflib is never given something it
could dereference, and an explicit format means a content-negotiated HTML
response can never surface as a confusing ``ParserError``.

**rdflib does not follow ``owl:imports``.** ``Graph.parse`` loads exactly one
document. Resolving the closure is our job (:func:`load_closure`), against a
LOCAL catalog built by parsing the files under the source folders, with a cycle
guard — ``A imports B imports A`` is normal in ontology corpora.
``owl:versionIRI`` is recorded first, because "which ontology is this" is a
property of the closure and not of any single file.

**Only some hierarchy predicates are transitive.** ``rdfs:subClassOf``,
``rdfs:subPropertyOf`` and ``skos:broaderTransitive`` compose;
``skos:broader``/``skos:narrower`` do NOT. :func:`subsumption_closure`
materialises the closure of the first group with one SPARQL property path — no
reasoner, no new dependency — and its callers mark those edges ``inferred``.
``skos:broader`` is deliberately absent from it: composing it is the exact defect
this module's callers guard against.

**Blank node identifiers are not stable across parses.** rdflib mints a fresh
id every time a document is read, so nothing may be keyed on one.
:func:`skolem_ids` mints deterministic ids from blank-node CONTENT instead.

A plain ``Graph`` is used throughout — one default graph, no quads. If named
graphs are ever needed the type to reach for is ``Dataset``;
``ConjunctiveGraph`` is deprecated in rdflib 7 and goes away in 8.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("graphknows.symbolic.ontology.rdf_io")

# Suffix -> rdflib format. Explicit, never guessed: ".owl" is RDF/XML and rdflib
# does not infer it, and passing `format=` is what keeps a mis-typed source from
# arriving as a parser error about the wrong syntax.
RDF_FORMATS: dict[str, str] = {
    ".ttl": "turtle",
    ".owl": "xml",
    ".rdf": "xml",
    ".xml": "xml",
    ".jsonld": "json-ld",
    ".json": "json-ld",
    ".nt": "nt",
    ".n3": "n3",
}

# OWL syntaxes rdflib has NO parser for. Naming the converter is the whole point
# of the error: "could not parse" sends someone hunting for a corrupt file.
_NEEDS_CONVERSION: dict[str, str] = {
    ".ofn": "OWL functional syntax",
    ".omn": "OWL Manchester syntax",
    ".obo": "OBO flat-file format",
}

# OASIS import catalogs parse as XML but carry no ontology content, and every
# OWL corpus ships one next to its modules.
SKIP_NAMES: tuple[str, ...] = ("catalog-",)

# URI schemes that name something to FETCH. A local path never has one (a
# Windows drive letter is a single character, so "c:/x" cannot collide).
_REMOTE_SCHEMES = frozenset({"http", "https", "ftp", "ftps", "file", "urn", "data"})

# Predicates that are transitive as asserted, for `subsumption_closure`.
_TRANSITIVE_HIERARCHY: tuple[str, ...] = (
    "rdfs:subClassOf",
    "rdfs:subPropertyOf",
    "skos:broaderTransitive",
)

_CLOSURE_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?s ?o WHERE {{
    ?s {predicate}+ ?o .
    FILTER (isIRI(?s) && isIRI(?o) && ?s != ?o)
}}
"""

_SKOLEM_PREFIX = "urn:graphknows:skolem:"


def new_graph() -> Any:
    """A fresh rdflib ``Graph``, naming the extra when rdflib is absent."""
    try:
        import rdflib
    except ImportError as exc:
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("Parsing RDF", "ontology") from exc
    return rdflib.Graph()


def local_path(source: str | Path) -> Path:
    """*source* as a local path, refusing anything that names a network location.

    Args:
        source: A path-like ontology source.

    Returns:
        The same source as a ``Path``.

    Raises:
        ValueError: ``source`` carries a URI scheme. Ontologies are read from
            disk here and never fetched, so a URL is a mistake worth naming
            rather than a FileNotFoundError about a mangled path.
    """
    text = str(source)
    if text.split(":", 1)[0].lower() in _REMOTE_SCHEMES:
        raise ValueError(
            f"Ontology sources are read from local files only, never fetched: {text}. "
            "Download the document first and pass the file (or the folder holding it)."
        )
    return Path(source)


def _unsupported(path: Path) -> str:
    """Why ``path`` cannot be parsed, naming the converter when one exists."""
    if syntax := _NEEDS_CONVERSION.get(path.suffix.lower()):
        return (
            f"{path.name} is {syntax}, which rdflib cannot parse. Convert it to Turtle "
            f"or RDF/XML first with ROBOT: "
            f"robot convert --input {path.name} --output {path.stem}.ttl"
        )
    return (
        f"Unsupported ontology format '{path.suffix or path.name}': {path}. "
        f"Expected one of {', '.join(sorted(RDF_FORMATS))}."
    )


def parse_file(path: Path, graph: Any = None) -> Any:
    """Parse ONE local RDF document into ``graph`` (a fresh one when omitted).

    The offline guard lives here and nowhere else: a ``Path`` is required, it is
    opened, and rdflib receives the open file plus an explicit format. rdflib
    therefore never holds a string it could resolve over the network.

    Args:
        path: The document to read. Must be a ``pathlib.Path``.
        graph: An existing graph to add the triples to.

    Returns:
        The graph the triples went into.

    Raises:
        TypeError: ``path`` is not a ``Path`` — the structural half of the guard.
        ValueError: ``path`` is a URL, or its suffix names no RDF syntax.
        FileNotFoundError: ``path`` does not exist.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"parse_file takes a pathlib.Path, not {type(path).__name__}: rdflib resolves a "
            "string source over the network, and ontology loading here is offline by design."
        )
    local = local_path(path)
    fmt = RDF_FORMATS.get(local.suffix.lower())
    if fmt is None:
        raise ValueError(_unsupported(local))
    if graph is None:
        graph = new_graph()
    with local.open("rb") as handle:
        graph.parse(file=handle, format=fmt, publicID=local.resolve().as_uri())
    return graph


@dataclass
class Closure:
    """One ``owl:imports`` closure, resolved against local files only."""

    graph: Any
    files: tuple[Path, ...]
    # Ontology IRI -> owl:versionIRI ("" when the document declares none). The
    # closure's identity, recorded before anything is asked of its contents.
    version_iris: dict[str, str]
    # Imported IRIs no local file answers to. Reported, never swallowed: a
    # missing import is why a domain/range silently resolves to nothing.
    unresolved: tuple[str, ...]


def _catalog_files(folders: Iterable[Path]) -> list[Path]:
    """Every RDF document under ``folders``, sorted, digests excluded.

    ``.json`` is our own digest suffix in this package — a digest is a parse
    RESULT, never an importable ontology document, and json-ld-parsing a 1 MB
    digest to learn that costs seconds.
    """
    return sorted(
        {
            path.resolve()
            for folder in folders
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in RDF_FORMATS
            and path.suffix != ".json"
            and not path.name.startswith(SKIP_NAMES)
        }
    )


def _declared_iris(graph: Any) -> set[str]:
    """Every IRI a document answers to: ``owl:Ontology`` subjects and their versions."""
    from rdflib import OWL, RDF

    out: set[str] = set()
    for onto in graph.subjects(RDF.type, OWL.Ontology):
        out.add(str(onto))
        out.update(str(version) for version in graph.objects(onto, OWL.versionIRI))
    return out


def _version_iris(graph: Any) -> dict[str, str]:
    """Ontology IRI -> ``owl:versionIRI``, empty string when undeclared."""
    from rdflib import OWL, RDF

    return {
        str(onto): str(next(iter(graph.objects(onto, OWL.versionIRI)), ""))
        for onto in graph.subjects(RDF.type, OWL.Ontology)
    }


def _imports(graph: Any) -> list[str]:
    """The IRIs this document imports."""
    from rdflib import OWL

    return sorted({str(target) for target in graph.objects(None, OWL.imports)})


def _catalog(folders: Iterable[Path]) -> tuple[dict[str, Path], dict[Path, Any]]:
    """Ontology IRI -> local file, plus the graph each file parsed into.

    rdflib ships no catalog and an ontology IRI is not derivable from a filename,
    so the only way to learn what a local document DECLARES is to parse it. The
    parsed graphs are kept, so resolving the closure never parses a file twice.

    ponytail: every RDF file under the search folders is parsed once. That is
    fine at install time (the bundled corpus is ~40 files) and would not be
    inside a request. If a corpus ever makes it slow, read the OASIS
    ``catalog-v001.xml`` that ships beside most of them instead of scanning.
    """
    index: dict[str, Path] = {}
    graphs: dict[Path, Any] = {}
    for path in _catalog_files(folders):
        try:
            graphs[path] = parse_file(path)
        except Exception as exc:
            # Counted, not hidden: a folder of assets legitimately mixes RDF
            # with junk, but "the import did not resolve" must stay explainable.
            log.info("catalog: %s is not usable RDF (%s: %s)", path, type(exc).__name__, exc)
            continue
        for iri in _declared_iris(graphs[path]):
            index.setdefault(iri, path)
    return index, graphs


def load_closure(entries: Sequence[Path], search: Sequence[Path] = ()) -> Closure:
    """Parse *entries* plus everything they ``owl:imports``, from local files only.

    Args:
        entries: Documents to start from.
        search: Folders to resolve imports against. Defaults to the folders the
            entries themselves live in.

    Returns:
        The merged :class:`Closure`, with the version IRIs it was built from and
        the imports no local file could answer.
    """
    files = [local_path(path) for path in entries]
    folders = list(search) or sorted({p if p.is_dir() else p.parent for p in files})
    index, parsed = _catalog(folders)

    graph = new_graph()
    versions: dict[str, str] = {}
    unresolved: set[str] = set()
    seen: set[Path] = set()
    queue: deque[Path] = deque(files)
    while queue:
        path = queue.popleft().resolve()
        if path in seen:
            continue  # the cycle guard: A imports B imports A terminates here
        seen.add(path)
        document = parsed.get(path) or parse_file(path)
        versions.update(_version_iris(document))  # identity first, contents after
        graph += document
        for target in _imports(document):
            resolved = index.get(target)
            if resolved is None:
                unresolved.add(target)
            else:
                queue.append(resolved)

    if unresolved:
        log.warning(
            "owl:imports unresolved locally (%d): %s — terms they define will not resolve",
            len(unresolved),
            ", ".join(sorted(unresolved)),
        )
    # Sorted, not BFS order: `version_iris` is serialised straight into a digest,
    # and BFS order is the order the ENTRIES were passed. Unsorted, the same
    # corpus digested with `--source` in a different order produced a
    # byte-different file whose terms were identical — which is enough to defeat
    # any "is the shipped asset what its sources produce" check downstream.
    return Closure(
        graph, tuple(sorted(seen)), dict(sorted(versions.items())), tuple(sorted(unresolved))
    )


def subsumption_closure(graph: Any) -> list[tuple[str, str]]:
    """``(child IRI, ancestor IRI)`` pairs over the predicates that ARE transitive.

    One SPARQL property path per predicate, which is why this needs no reasoner
    and no ``owlrl`` dependency. ``skos:broader`` is NOT among them: it is not
    transitive, and a scheme that wants its closure asserts
    ``skos:broaderTransitive``, which is.

    ponytail: SKOS entails ``broader ⊑ broaderTransitive``, so a strict reasoner
    WOULD compose a broader chain into broaderTransitive links. Not materialising
    them keeps the asserted and inferred hierarchies honest here; add
    ``skos:broader`` to ``_TRANSITIVE_HIERARCHY`` if a consumer ever needs the
    full SKOS entailment, and it will land marked ``inferred`` like the rest.
    """
    out: set[tuple[str, str]] = set()
    for predicate in _TRANSITIVE_HIERARCHY:
        for row in graph.query(_CLOSURE_QUERY.format(predicate=predicate)):
            out.add((str(row[0]), str(row[1])))
    return sorted(out)


def skolem_ids(graph: Any, *, max_depth: int = 8) -> dict[str, str]:
    """Blank node id -> a deterministic IRI, stable across parses of the same file.

    rdflib mints a fresh ``BNode`` id on every parse, so keying a vertex, a cache
    or a stored block on one produces a different answer each run. The id here is
    a hash of what the blank node SAYS — its predicate/object pairs, recursively
    for nested blank nodes, plus the IRI subjects that point at it — so the same
    axiom in the same document always mints the same id.

    ponytail: two structurally identical blank nodes under the same IRI anchor
    share an id (they are the same axiom asserted twice, so collapsing them is
    right), and a cycle between blank nodes hashes as a marker rather than as its
    shape. The complete answer is ``rdflib.compare.to_isomorphic``'s canonical
    labelling; it exposes no public blank-node -> label map (only canonical
    TRIPLES) and its colour refinement is superlinear on pathological graphs.
    Upgrade there if a corpus ever needs isomorphic-but-differently-anchored
    blank nodes told apart.

    Args:
        graph: The parsed graph.
        max_depth: How deep a nested blank-node expression is followed.

    Returns:
        Blank node id (``str(bnode)``) -> skolem IRI.
    """
    from rdflib import BNode

    def content(node: Any, depth: int, seen: frozenset[str]) -> str:
        if not isinstance(node, BNode):
            return str(node.n3())
        key = str(node)
        if key in seen or depth > max_depth:
            return "<bounded>"
        inner = seen | {key}
        anchors = sorted(
            f"{subject.n3()} {pred.n3()}"
            for subject, pred in graph.subject_predicates(node)
            if not isinstance(subject, BNode)
        )
        says = sorted(
            f"{pred.n3()} {content(obj, depth + 1, inner)}"
            for pred, obj in graph.predicate_objects(node)
        )
        return "[ " + " ; ".join([*anchors, *says]) + " ]"

    return {
        str(node): _SKOLEM_PREFIX
        + hashlib.sha256(content(node, 0, frozenset()).encode("utf-8")).hexdigest()[:24]
        for node in graph.all_nodes()
        if isinstance(node, BNode)
    }
