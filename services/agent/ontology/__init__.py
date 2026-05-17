"""Compatibility shim — use ``graphknows.shared.ontology`` instead."""
from graphknows.shared.ontology import (  # noqa: F401
    OntologyLoader,
    OntologyManifest,
    OntologyRelExtractor,
    OntologyRouter,
)

__all__ = ["OntologyLoader", "OntologyManifest", "OntologyRelExtractor", "OntologyRouter"]
