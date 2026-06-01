"""Ontology sub-package — loader, router, relation extractor, and label resolution.

Example:
    from graphknows.ingestion.ontology import OntologyLoader, OntologyRouter
"""

from graphknows.ingestion.ontology.labels import DEFAULT_LABELS, resolve_entity_labels
from graphknows.ingestion.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ingestion.ontology.ports import OntologyProvider
from graphknows.ingestion.ontology.rdf_parser import OntologyLabelExtractor, extract_ontology_labels
from graphknows.ingestion.ontology.registry import ManifestOntologyProvider, OntologyRegistry
from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor
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
