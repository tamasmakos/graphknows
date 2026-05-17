"""Ontology subpackage — loader, router, and relation extractor."""

from .loader import OntologyLoader, OntologyManifest
from .rel_extractor import OntologyRelExtractor
from .router import OntologyRouter

__all__ = ["OntologyLoader", "OntologyManifest", "OntologyRelExtractor", "OntologyRouter"]
