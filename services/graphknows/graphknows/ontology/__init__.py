"""Ontology sub-package — loader, router, and relation extractor.

Example:
    from graphknows.ontology import OntologyLoader, OntologyRouter
"""

from graphknows.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.ontology.rel_extractor import OntologyRelExtractor
from graphknows.ontology.router import OntologyRouter

__all__ = ["OntologyLoader", "OntologyManifest", "OntologyRelExtractor", "OntologyRouter"]
