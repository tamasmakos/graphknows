"""Ontology router — selects the best ontology for a given document text."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from .loader import OntologyLoader, OntologyManifest

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/text-embedding-3-small"
_OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


def _embed_via_openrouter(texts: list[str]) -> np.ndarray | None:
    """Embed texts via OpenRouter. Returns (N, D) float32 or None on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or not texts:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                _OPENROUTER_EMBED_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": _DEFAULT_MODEL, "input": texts},
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            return np.array([d["embedding"] for d in data], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("OntologyRouter: OpenRouter embed failed (%s) — keyword-only mode", exc)
        return None


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
    ) -> None:
        self._loader = loader or OntologyLoader(ontology_root)
        self._model_name = model_name
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
        descriptions = [m.description for m in self._manifests]
        embs = _embed_via_openrouter(descriptions)
        if embs is not None:
            norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
            self._desc_embeddings = embs / norms
            log.info(
                "OntologyRouter ready with %d ontologies (embeddings OK)", len(self._manifests)
            )
        else:
            self._desc_embeddings = None
            log.info(
                "OntologyRouter ready with %d ontologies (keyword-only mode)", len(self._manifests)
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
        if self._desc_embeddings is None:
            return [0.0] * len(self._manifests)
        try:
            doc_emb = _embed_via_openrouter([text[:512]])
            if doc_emb is None:
                return [0.0] * len(self._manifests)
            doc_norm = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True).clip(min=1e-9)
            sims = (self._desc_embeddings @ doc_norm.T).flatten().tolist()
            return [float(s) for s in sims]
        except Exception as exc:
            log.warning("Embedding score failed: %s", exc)
            return [0.0] * len(self._manifests)

    def select(self, text: str, alpha: float = 0.4) -> OntologyManifest:
        """Select the best ontology for a document.

        Args:
            text: Document text (first 2000 chars used for routing).
            alpha: Weight for keyword score vs. embedding score (0–1).
                   1.0 = pure keywords, 0.0 = pure embeddings.

        Returns:
            The OntologyManifest with the highest combined score.
        """
        self._ensure_ready()
        if not self._manifests:
            raise RuntimeError("No ontologies available for routing")

        snippet = text[:2000]
        kw_scores = self._keyword_scores(snippet)
        emb_scores = self._embedding_scores(snippet)

        combined = [alpha * kw + (1 - alpha) * emb for kw, emb in zip(kw_scores, emb_scores)]
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

    def select_multi(self, text: str, top_k: int = 2, alpha: float = 0.4) -> list[OntologyManifest]:
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

        snippet = text[:2000]
        kw_scores = self._keyword_scores(snippet)
        emb_scores = self._embedding_scores(snippet)
        combined = [alpha * kw + (1 - alpha) * emb for kw, emb in zip(kw_scores, emb_scores)]
        indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
        return [self._manifests[i] for i in indices]
