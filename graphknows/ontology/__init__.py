"""Ontology support — RDF/OWL term loading and definition-embedding matching.

A top-level package rather than part of ``ingestion`` because both halves of the
system use it: ingestion steers entity and relation extraction with ontology
labels, and retrieval matches queries against the same term definitions. It is
also mode-independent — ontology injection works identically in ``llm_free`` and
``llm_assisted``.

The load-bearing pair is :func:`~graphknows.ontology.loader.load_ontology_terms`
(classes and properties *with their definitions*, from a file or a folder) and
:func:`~graphknows.ontology.catalog.load_ontology_index` (those definitions
embedded once, disk-cached, cosine-matchable against a chunk or query embedding).

An ontology is always addressed by **path** — ``GRAPHKNOWS_ONTOLOGY`` takes a
file or a folder. Bundled ontologies live in ``assets/`` and are referenced the
same way; there is no by-name registry.

These imports stay LLM-free. DSPy-backed relation extraction lives in
``graphknows.ingestion.extraction.relations.llm_assisted.dspy``.
"""

from graphknows.ontology.catalog import (
    DEFAULT_MIN_SIM,
    DEFAULT_TOP_K,
    TAG_MIN_SIM,
    OntologyIndex,
    load_ontology_index,
    reset_ontology_indexes,
)
from graphknows.ontology.loader import (
    OntologyTerm,
    RdfOntologyManifest,
    load_ontology_terms,
    load_rdf_ontology_file,
)

__all__ = [
    "DEFAULT_MIN_SIM",
    "DEFAULT_TOP_K",
    "TAG_MIN_SIM",
    "OntologyIndex",
    "OntologyTerm",
    "RdfOntologyManifest",
    "load_ontology_index",
    "load_ontology_terms",
    "load_rdf_ontology_file",
    "reset_ontology_indexes",
]
