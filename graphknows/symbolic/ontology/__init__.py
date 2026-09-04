"""Ontology support — RDF/OWL term loading and definition-embedding matching.

A top-level package rather than part of ``ingestion`` because both halves of the
system use it: ingestion steers entity and relation extraction with ontology
labels, and retrieval matches queries against the same term definitions. It is
also mode-independent — ontology injection works identically in ``llm_free`` and
``llm_assisted``.

The load-bearing pair is :func:`~graphknows.symbolic.ontology.loader.load_ontology_terms`
(classes and properties *with their definitions*, from a file or a folder) and
:func:`~graphknows.symbolic.ontology.catalog.load_ontology_index` (those definitions
embedded once, disk-cached, cosine-matchable against a chunk or query embedding).

An ontology is always addressed by **path** — ``GRAPHKNOWS_ONTOLOGY`` takes a
file or a folder. Bundled ontologies live in ``assets/`` and are referenced the
same way; there is no by-name registry.

Bring your own ontology with ``python -m graphknows.symbolic.ontology.digest`` (see
:mod:`graphknows.symbolic.ontology.digest`), then point ``GRAPHKNOWS_ONTOLOGY`` at the
result.

These imports stay LLM-free. The LLM decoder lives in
``graphknows.ingestion.extraction.llm``.

This facade carries only what is imported from outside the package; everything
else is reached through its own module, so adding a name here is a deliberate
act of making it public.
"""

from graphknows.symbolic.ontology.catalog import (
    DEFAULT_MIN_SIM,
    TAG_MIN_SIM,
    OntologyIndex,
    load_ontology_index,
)
from graphknows.symbolic.ontology.labels import COARSE, entity_labels, relation_labels
from graphknows.symbolic.ontology.lexical import evoked, usable_index
from graphknows.symbolic.ontology.loader import (
    OntologyTerm,
    RdfOntologyManifest,
    load_ontology_terms,
    load_rdf_ontology_file,
)
from graphknows.symbolic.ontology.skos import Concept, ancestors

__all__ = [
    "COARSE",
    "DEFAULT_MIN_SIM",
    "TAG_MIN_SIM",
    "Concept",
    "OntologyIndex",
    "OntologyTerm",
    "RdfOntologyManifest",
    "ancestors",
    "entity_labels",
    "evoked",
    "load_ontology_index",
    "load_ontology_terms",
    "load_rdf_ontology_file",
    "relation_labels",
    "usable_index",
]
