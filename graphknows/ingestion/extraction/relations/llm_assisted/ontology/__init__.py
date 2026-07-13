"""Ontology helpers — RDF/OWL metadata loading, routing, and registry.

These imports stay LLM-free. DSPy-backed relation extraction lives in
``graphknows.ingestion.extraction.relations.llm_assisted.dspy`` and is wired via the config adapters.
"""

from graphknows.ingestion.extraction.relations.llm_assisted.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ingestion.extraction.relations.llm_assisted.ontology.ports import OntologyProvider
from graphknows.ingestion.extraction.relations.llm_assisted.ontology.rdf_parser import OntologyLabelExtractor, extract_ontology_labels
from graphknows.ingestion.extraction.relations.llm_assisted.ontology.registry import ManifestOntologyProvider, OntologyRegistry
from graphknows.ingestion.extraction.relations.llm_assisted.ontology.router import OntologyRouter

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
