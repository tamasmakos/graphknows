"""Bring your own RDF: project arbitrary RDF/OWL/SKOS onto the graph and a digest.

:mod:`graphknows.ontology.digest` is the small door — one ontology, one JSON
file, CCO's conventions assumed. This subpackage is the large one: a *folder* of
mixed RDF, several releases of the same vocabulary, typo'd hosts, unusable
document prefixes, and defined classes whose only parent hides inside an
``owl:equivalentClass``. It lands two layers in a namespace (``ONODE`` faithful
to the IRIs, ``ONTOLOGY_CLASS`` label-keyed and hierarchical) and can export the
same taxonomy as a standard digest for a base install to read back.

Run it with ``python -m graphknows.ontology.rdf``. Needs the ``assisted`` extra.

Each stage is separately usable and separately testable: :mod:`prefixes` is pure
stdlib, :mod:`taxonomy` is pure data-in/data-out, and only :mod:`store` talks to
a database.
"""

from graphknows.ontology.rdf.digest_export import to_digest
from graphknows.ontology.rdf.prefixes import BOOTSTRAP, Prefixes, split_iri
from graphknows.ontology.rdf.projection import NodeMeta, PredInfo, load_dir, project
from graphknows.ontology.rdf.store import (
    ensure_schema,
    write_projection,
    write_taxonomy,
)
from graphknows.ontology.rdf.taxonomy import Taxon, extract

__all__ = [
    "BOOTSTRAP",
    "NodeMeta",
    "PredInfo",
    "Prefixes",
    "Taxon",
    "ensure_schema",
    "extract",
    "load_dir",
    "project",
    "split_iri",
    "to_digest",
    "write_projection",
    "write_taxonomy",
]
