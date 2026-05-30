"""Ontology sub-package — loader, router, relation extractor, and label resolution.

Example:
    from graphknows.ontology import OntologyLoader, OntologyRouter
"""

from graphknows.ontology.labels import DEFAULT_LABELS, resolve_entity_labels
from graphknows.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ontology.rdf_parser import OntologyLabelExtractor, extract_ontology_labels
from graphknows.ontology.rel_extractor import OntologyRelExtractor
from graphknows.ontology.router import OntologyRouter

__all__ = [
    "DEFAULT_LABELS",
    "OntologyLabelExtractor",
    "OntologyLoader",
    "OntologyManifest",
    "OntologyRelExtractor",
    "OntologyRouter",
    "extract_ontology_labels",
    "resolve_entity_labels",
]
