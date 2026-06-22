"""Registry for selecting ontology providers by document type or content."""

from __future__ import annotations

from dataclasses import dataclass

from graphknows.extraction.ontology.loader import OntologyLoader, OntologyManifest
from graphknows.extraction.ontology.ports import OntologyProvider


@dataclass(frozen=True)
class ManifestOntologyProvider:
    """OntologyProvider backed by an OntologyManifest."""

    manifest: OntologyManifest

    @property
    def name(self) -> str:
        """Return the manifest name."""
        return self.manifest.name

    def supports(self, doc_type: str | None = None, text: str = "") -> bool:
        """Return True when doc_type or text keywords match this manifest."""
        candidates = {item.lower() for item in self.manifest.domains}
        if doc_type and doc_type.lower() in candidates:
            return True
        lower = text.lower()
        return any(keyword.lower() in lower for keyword in self.manifest.keywords)

    def class_labels(self) -> list[str]:
        """Return class labels from the manifest."""
        return list(self.manifest.class_labels)

    def property_labels(self) -> list[str]:
        """Return property labels from the manifest."""
        return list(self.manifest.property_labels)

    def relation_types(self) -> list[str]:
        """Return relation labels from the manifest."""
        return list(self.manifest.property_labels)

    def summary_jsonld(self) -> str:
        """Return the compact JSON-LD summary from the manifest."""
        return self.manifest.summary_jsonld


class OntologyRegistry:
    """Interface-driven registry for ontology providers."""

    def __init__(self, providers: list[OntologyProvider] | None = None) -> None:
        self._providers = list(providers or [])

    @classmethod
    def from_loader(cls, loader: OntologyLoader | None = None) -> OntologyRegistry:
        """Build a registry from bundled ontology manifests."""
        ontology_loader = loader or OntologyLoader()
        providers = [ManifestOntologyProvider(manifest) for manifest in ontology_loader.load_all()]
        return cls(providers)

    def register(self, provider: OntologyProvider) -> None:
        """Register an ontology provider."""
        self._providers.append(provider)

    def providers(self) -> list[OntologyProvider]:
        """Return registered providers."""
        return list(self._providers)

    def for_document(self, doc_type: str | None = None, text: str = "") -> OntologyProvider:
        """Select the first provider that supports the document."""
        for provider in self._providers:
            if provider.supports(doc_type=doc_type, text=text):
                return provider
        if self._providers:
            return self._providers[0]
        raise RuntimeError("No ontology providers registered")
