"""Ontology helpers with lazy relation-extractor loading.

Ontology utility imports stay LLM-free unless the caller explicitly requests the
DSPy-backed relation extractor.
"""

from typing import Any

from graphknows.ingestion.ontology.labels import DEFAULT_LABELS, resolve_entity_labels
from graphknows.ingestion.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ingestion.ontology.ports import OntologyProvider
from graphknows.ingestion.ontology.rdf_parser import OntologyLabelExtractor, extract_ontology_labels
from graphknows.ingestion.ontology.registry import ManifestOntologyProvider, OntologyRegistry
from graphknows.ingestion.ontology.router import OntologyRouter

__all__ = [
    "DEFAULT_LABELS",
    "ManifestOntologyProvider",
    "OntologyLabelExtractor",
    "OntologyLoader",
    "OntologyManifest",
    "OntologyProvider",
    "OntologyRegistry",
    "OntologyRelExtractor",
    "OntologyRouter",
    "extract_ontology_labels",
    "resolve_entity_labels",
]


def __getattr__(name: str) -> Any:
    if name == "OntologyRelExtractor":
        from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor

        return OntologyRelExtractor
    raise AttributeError(name)
