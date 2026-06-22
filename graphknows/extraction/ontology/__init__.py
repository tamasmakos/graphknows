"""Ontology helpers — RDF/OWL metadata loading, routing, and registry.

These imports stay LLM-free. DSPy-backed relation extraction lives in
``graphknows.extraction.relations.dspy`` and is wired via the config adapters.
"""

from graphknows.extraction.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.extraction.ontology.ports import OntologyProvider
from graphknows.extraction.ontology.rdf_parser import OntologyLabelExtractor, extract_ontology_labels
from graphknows.extraction.ontology.registry import ManifestOntologyProvider, OntologyRegistry
from graphknows.extraction.ontology.router import OntologyRouter

__all__ = [
    "ManifestOntologyProvider",
    "OntologyLabelExtractor",
    "OntologyLoader",
    "OntologyManifest",
    "OntologyProvider",
    "OntologyRegistry",
    "OntologyRouter",
    "extract_ontology_labels",
]
