"""Ontology loader — parses bundled ontology files and returns structured manifests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Root of the bundled ontology package
_ONTOLOGY_ROOT = Path(__file__).parent.parent.parent.parent / "packages" / "ontologies"
_REGISTRY_PATH = _ONTOLOGY_ROOT / "registry.json"


@dataclass
class OntologyManifest:
    """Structured view of a single ontology for use in extraction."""

    id: str
    name: str
    description: str
    domains: list[str]
    keywords: list[str]
    class_labels: list[str]
    property_labels: list[str]
    summary_jsonld: str  # compact JSON-LD string for LLM context


def _load_jsonld_manifest(summary_path: Path, meta: dict) -> OntologyManifest:
    """Build manifest from a JSON-LD summary file."""
    text = summary_path.read_text(encoding="utf-8")
    data = json.loads(text)
    classes: list[str] = data.get("classes", [])
    properties: list[str] = data.get("properties", [])
    return OntologyManifest(
        id=meta["id"],
        name=meta["name"],
        description=meta["description"],
        domains=meta["domains"],
        keywords=meta["keywords"],
        class_labels=classes,
        property_labels=properties,
        summary_jsonld=text,
    )


def _load_owl_manifest(file_paths: list[Path], meta: dict, summary_path: Path | None) -> OntologyManifest:
    """Build manifest from OWL/Turtle files using rdflib."""
    try:
        import rdflib  # type: ignore
        from rdflib import RDF, RDFS, OWL  # type: ignore

        g = rdflib.Dataset()
        for p in file_paths:
            if p.exists():
                try:
                    g.parse(str(p))
                except Exception as exc:
                    log.warning("Failed to parse %s: %s", p, exc)

        classes: list[str] = []
        properties: list[str] = []

        for s in g.subjects(RDF.type, OWL.Class):
            label = g.value(s, RDFS.label)
            if label:
                classes.append(str(label))
            elif "#" in str(s):
                classes.append(str(s).split("#")[-1])
            elif "/" in str(s):
                classes.append(str(s).split("/")[-1])

        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            label = g.value(s, RDFS.label)
            if label:
                properties.append(str(label))
            elif "#" in str(s):
                properties.append(str(s).split("#")[-1])

        # Also grab DataProperties
        for s in g.subjects(RDF.type, OWL.DatatypeProperty):
            label = g.value(s, RDFS.label)
            if label:
                properties.append(str(label))

        log.debug("Parsed %s: %d classes, %d properties", meta["id"], len(classes), len(properties))

    except ImportError:
        log.warning("rdflib not available — falling back to empty class list for %s", meta["id"])
        classes = []
        properties = []

    # Prefer the summary file for LLM context (compact), fall back to generated list
    if summary_path and summary_path.exists():
        summary_jsonld = summary_path.read_text(encoding="utf-8")
    else:
        summary_jsonld = json.dumps({
            "id": meta["id"],
            "classes": classes[:100],
            "properties": properties[:50],
        })

    return OntologyManifest(
        id=meta["id"],
        name=meta["name"],
        description=meta["description"],
        domains=meta["domains"],
        keywords=meta["keywords"],
        class_labels=classes or [],
        property_labels=properties or [],
        summary_jsonld=summary_jsonld,
    )


class OntologyLoader:
    """Load ontology manifests from the bundled registry."""

    def __init__(self, ontology_root: Path | None = None) -> None:
        self._root = ontology_root or _ONTOLOGY_ROOT
        self._registry: list[dict] = []
        self._cache: dict[str, OntologyManifest] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        reg_path = self._root / "registry.json"
        if not reg_path.exists():
            log.error("Ontology registry not found at %s", reg_path)
            return
        with reg_path.open(encoding="utf-8") as fh:
            self._registry = json.load(fh).get("ontologies", [])
        log.info("Loaded ontology registry with %d entries", len(self._registry))

    def list_ids(self) -> list[str]:
        """Return all ontology IDs in the registry."""
        return [o["id"] for o in self._registry]

    def load(self, ontology_id: str) -> OntologyManifest:
        """Load and return an OntologyManifest by ID.

        Args:
            ontology_id: Ontology identifier matching registry entry.

        Returns:
            OntologyManifest with class and property labels populated.

        Raises:
            KeyError: If ontology_id not in registry.
        """
        if ontology_id in self._cache:
            return self._cache[ontology_id]

        meta = next((o for o in self._registry if o["id"] == ontology_id), None)
        if meta is None:
            raise KeyError(f"Ontology '{ontology_id}' not found in registry")

        summary_file = meta.get("summary_file")
        summary_path = (self._root / summary_file) if summary_file else None

        fmt = meta.get("format", "json-ld")
        if fmt == "json-ld" and summary_path and summary_path.exists():
            manifest = _load_jsonld_manifest(summary_path, meta)
        else:
            file_paths = [self._root / f for f in meta.get("files", [])]
            manifest = _load_owl_manifest(file_paths, meta, summary_path)

        self._cache[ontology_id] = manifest
        return manifest

    def load_all(self) -> list[OntologyManifest]:
        """Load all ontologies in the registry."""
        return [self.load(oid) for oid in self.list_ids()]
