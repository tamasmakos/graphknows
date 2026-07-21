"""Ontology router — selects the best ontology for a given document text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .loader import OntologyLoader, OntologyManifest

if TYPE_CHECKING:
    from graphknows.storage.base import IEmbedder

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_ALPHA: float = 0.4  # keyword vs. embedding weight; 1.0 = pure keywords
_ROUTING_SNIPPET_CHARS: int = 2000


class OntologyRouter:
    """Route a document to the most relevant ontology.

    Uses a two-stage approach:
    1. Keyword frequency heuristic (fast, cheap).
    2. Embedding cosine similarity (BAAI/bge-small-en-v1.5).

    The ontology with the highest combined score wins.
    """

    def __init__(
        self,
        loader: OntologyLoader | None = None,
        model_name: str = _DEFAULT_MODEL,
        ontology_root: Path | None = None,
        embedder: IEmbedder | None = None,
    ) -> None:
        self._loader = loader or OntologyLoader(ontology_root)
        self._model_name = model_name
        self._embedder = embedder
        self._manifests: list[OntologyManifest] = []
        self._desc_embeddings: np.ndarray | None = None
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._manifests = self._loader.load_all()
        if not self._manifests:
            log.warning("OntologyRouter: no ontologies loaded — routing will always fail")
            self._ready = True
            return
        if self._embedder is not None:
            try:
                descriptions = [m.description for m in self._manifests]
                self._desc_embeddings = np.asarray(self._embedder.embed(descriptions))
                log.info("OntologyRouter ready with %d ontologies", len(self._manifests))
            except Exception as exc:
                log.warning(
                    "OntologyRouter: embedding model unavailable (%s) — keyword-only mode",
                    exc,
                )
                self._desc_embeddings = None
        else:
            log.info(
                "OntologyRouter ready in keyword-only mode with %d ontologies", len(self._manifests)
            )
        self._ready = True

    def _keyword_scores(self, text: str) -> list[float]:
        """Return normalised keyword frequency score per ontology."""
        lower = text.lower()
        words = set(lower.split())
        scores: list[float] = []
        for m in self._manifests:
            hits = sum(1 for kw in m.keywords if kw in lower or kw in words)
            scores.append(hits / max(len(m.keywords), 1))
        return scores

    def _embedding_scores(self, text: str) -> list[float]:
        """Return cosine similarity scores per ontology, or zeros if unavailable."""
        if self._embedder is None or self._desc_embeddings is None:
            return [0.0] * len(self._manifests)
        try:
            doc_emb = np.asarray(self._embedder.embed([text[:512]]))
            sims = (self._desc_embeddings @ doc_emb.T).flatten().tolist()
            return [float(s) for s in sims]
        except Exception as exc:
            log.warning("Embedding score failed: %s", exc)
            return [0.0] * len(self._manifests)

    def select(self, text: str, alpha: float = _DEFAULT_ALPHA) -> OntologyManifest:
        """Select the best ontology for a document.

        Args:
            text: Document text (first 2000 chars used for routing).
            alpha: Weight for keyword score vs. embedding score (0-1).
                   1.0 = pure keywords, 0.0 = pure embeddings.

        Returns:
            The OntologyManifest with the highest combined score.
        """
        self._ensure_ready()
        if not self._manifests:
            raise RuntimeError("No ontologies available for routing")

        snippet = text[:_ROUTING_SNIPPET_CHARS]
        kw_scores = self._keyword_scores(snippet)
        emb_scores = self._embedding_scores(snippet)

        combined = [
            alpha * kw + (1 - alpha) * emb for kw, emb in zip(kw_scores, emb_scores, strict=False)
        ]
        best_idx = int(np.argmax(combined))
        best = self._manifests[best_idx]
        log.debug(
            "OntologyRouter selected '%s' (kw=%.3f emb=%.3f combined=%.3f)",
            best.id,
            kw_scores[best_idx],
            emb_scores[best_idx],
            combined[best_idx],
        )
        return best

    def select_rdf_file(self, text: str, alpha: float = _DEFAULT_ALPHA) -> str | None:
        """Select the best ontology and return its primary RDF file path, if any.

        This is a convenience wrapper that combines :meth:`select` with a
        registry lookup so callers can directly feed the result into
        :func:`~graphknows.ontology.loader.load_rdf_ontology_file`.

        Returns:
            Absolute path string to the RDF/OWL/Turtle file, or ``None`` if the
            selected ontology has no file entries in the registry.
        """
        self._ensure_ready()
        manifest = self.select(text, alpha=alpha)
        # The OntologyLoader internal registry holds the raw dicts; look up files.
        raw_entries = self._loader._registry  # type: ignore[attr-defined]
        entry = next((e for e in raw_entries if e.get("id") == manifest.id), None)
        if not entry or not entry.get("files"):
            return None
        first_file = entry["files"][0]
        root = self._loader._root  # type: ignore[attr-defined]
        return str(root / first_file)

    def select_multi(
        self, text: str, top_k: int = 2, alpha: float = _DEFAULT_ALPHA
    ) -> list[OntologyManifest]:
        """Select top-k ontologies for multi-domain documents.

        Args:
            text: Document text.
            top_k: Number of ontologies to return.
            alpha: Keyword weight (see :meth:`select`).

        Returns:
            List of OntologyManifest sorted by descending score.
        """
        self._ensure_ready()
        if not self._manifests:
            return []

        snippet = text[:_ROUTING_SNIPPET_CHARS]
        kw_scores = self._keyword_scores(snippet)
        emb_scores = self._embedding_scores(snippet)
        combined = [
            alpha * kw + (1 - alpha) * emb for kw, emb in zip(kw_scores, emb_scores, strict=False)
        ]
        indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
        return [self._manifests[i] for i in indices]
