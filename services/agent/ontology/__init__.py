"""Ontology subpackage — loader, router, and relation extractor."""

from .loader import OntologyLoader, OntologyManifest
from .router import OntologyRouter

__all__ = ["OntologyLoader", "OntologyManifest", "OntologyRouter"]
