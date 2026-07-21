"""Ontology Label Extractor.

Parses RDF/OWL ontology files to extract class labels for use as
GLiNER entity extraction labels. This enables domain-specific entity
recognition based on ontology class definitions.

Usage:
    extractor = OntologyLabelExtractor("/app/input/ontology/cdm-4.13.2")
    labels = extractor.extract_labels()
"""

import logging
import re
from pathlib import Path
from typing import cast

try:
    from rdflib import BNode, Graph, Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS
except ImportError as exc:
    raise ImportError("rdflib is required. Install with: pip install rdflib") from exc

log = logging.getLogger(__name__)


class OntologyLabelExtractor:
    """Extracts class labels from RDF/OWL ontology files.

    This class parses RDF files from a specified directory and extracts
    rdfs:label values from OWL Classes and RDFS Classes. The extracted
    labels can be used for entity extraction systems like GLiNER.

    Attributes:
        ontology_dir: Path to directory containing RDF files.
        namespace_filter: Optional namespace prefix to filter classes.
        graph: RDFLib graph containing loaded ontology triples.
    """

    def __init__(self, ontology_dir: str, namespace_filter: str | None = None):
        """Initialize the extractor.

        Args:
            ontology_dir: Path to directory containing RDF/OWL files.
            namespace_filter: Optional namespace URI prefix to filter classes.
                             Only classes starting with this prefix will be included.
        """
        self.ontology_dir = Path(ontology_dir)
        self.namespace_filter = namespace_filter
        self.graph = Graph()
        self._loaded = False

    def _load_rdf_files(self) -> int:
        """Load all RDF files from the ontology directory.

        Returns:
            Number of files successfully loaded.
        """
        if self._loaded:
            return 0

        if not self.ontology_dir.exists():
            log.warning("Ontology directory not found: %s", self.ontology_dir)
            return 0

        rdf_files = list(self.ontology_dir.glob("*.rdf"))
        if not rdf_files:
            log.warning("No RDF files found in %s", self.ontology_dir)
            return 0

        loaded_count = 0
        for rdf_file in rdf_files:
            try:
                self.graph.parse(rdf_file, format="xml")
                loaded_count += 1
            except Exception as e:
                log.warning("Could not parse %s: %s", rdf_file.name, e)

        self._loaded = True
        log.info("Loaded %d RDF files with %d triples", loaded_count, len(self.graph))
        return loaded_count

    def _get_label(self, uri: URIRef) -> str | None:
        """Get the rdfs:label for a URI, preferring English labels.

        Args:
            uri: The URI to get the label for.

        Returns:
            The label string, or None if no label found.
        """
        labels = list(self.graph.objects(uri, RDFS.label))
        if not labels:
            return None

        # Prefer English labels
        for label in labels:
            if isinstance(label, Literal) and (label.language == "en" or not label.language):
                return str(label)

        # Fallback to first available label
        return str(labels[0])

    def _extract_local_name(self, uri: str) -> str:
        """Extract the local name from a URI.

        Args:
            uri: Full URI string.

        Returns:
            The local name (fragment or last path segment).
        """
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.split("/")[-1]

    def extract_labels(
        self, include_local_names: bool = True, top_level_only: bool = True, min_subclasses: int = 0
    ) -> list[str]:
        """Extract class labels from the loaded ontology with hierarchical filtering.

        Args:
            include_local_names: If True, use URI local names as fallback.
            top_level_only: If True, only include classes with no named parents.
            min_subclasses: only include classes with at least this many subclasses.

        Returns:
            List of unique class labels.
        """
        self._load_rdf_files()

        labels: set[str] = set()

        # Get all OWL and RDFS Classes
        all_classes = set(self.graph.subjects(RDF.type, OWL.Class)).union(
            set(self.graph.subjects(RDF.type, RDFS.Class))
        )

        log.info("Analyzing %d classes in ontology", len(all_classes))

        # Technical classes to exclude
        exclude_classes = {
            OWL.Thing,
            OWL.Nothing,
            OWL.NamedIndividual,
            RDFS.Resource,
            RDFS.Class,
            OWL.Class,
        }

        for cls in all_classes:
            if isinstance(cls, BNode) or cls in exclude_classes:
                continue

            cls_uri = str(cls)

            # Apply namespace filter if specified
            if self.namespace_filter and not cls_uri.startswith(self.namespace_filter):
                continue

            # Hierarchical filtering
            parents = list(self.graph.objects(cls, RDFS.subClassOf))
            # Filter parents: ignore blank nodes and technical classes
            named_parents = [
                p for p in parents if isinstance(p, URIRef) and p not in exclude_classes
            ]

            if top_level_only and named_parents:
                continue

            if min_subclasses > 0:
                children = list(self.graph.subjects(RDFS.subClassOf, cls))
                if len(children) < min_subclasses:
                    continue

            # Try to get rdfs:label
            label = self._get_label(cast("URIRef", cls))

            if label:
                label = label.strip()
                if label and len(label) > 1:
                    labels.add(label)
            elif include_local_names:
                local_name = self._extract_local_name(cls_uri)
                if local_name and len(local_name) > 1:
                    readable = self._format_local_name(local_name)
                    # Filter out purely numerical or technical-looking local names
                    if any(c.isalpha() for c in readable):
                        labels.add(readable)

        result = sorted(labels)
        log.info(
            "Extracted %d unique labels from ontology (top_level=%s, min_children=%d)",
            len(result),
            top_level_only,
            min_subclasses,
        )
        return result

    def _format_local_name(self, name: str) -> str:
        """Format a local name into a readable label.

        Converts camelCase, snake_case, and hyphen-case to space-separated words.
        """
        # Replace underscores and hyphens with spaces
        name = name.replace("_", " ").replace("-", " ")

        # Insert space before uppercase letters (camelCase)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

        # Title case and collapse multiple spaces
        name = " ".join(name.split()).title()

        return name


def extract_ontology_labels(
    ontology_dir: str, namespace_filter: str | None = None, include_local_names: bool = True
) -> list[str]:
    """Convenience function to extract labels from an ontology directory.

    Args:
        ontology_dir: Path to directory containing RDF/OWL files.
        namespace_filter: Optional namespace prefix to filter classes.
        include_local_names: Use URI local names as fallback when no label.

    Returns:
        List of unique class labels.
    """
    extractor = OntologyLabelExtractor(ontology_dir, namespace_filter)
    return extractor.extract_labels(include_local_names)
