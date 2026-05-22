"""Centralised embedding model — OpenRouter-backed, no sentence-transformers."""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

from graphknows.retrieval.stores.embedder import EMBED_DIM, embed

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton embedding model backed by the shared OpenRouter embedder."""

    _instance: EmbeddingModel | None = None

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        logger.info("EmbeddingModel initialised (OpenRouter text-embedding-3-small)")

    @classmethod
    def get_instance(cls) -> EmbeddingModel:
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return EMBED_DIM

    @property
    def is_available(self) -> bool:
        """Always true — degrades gracefully to zero vectors when key missing."""
        return True

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,  # noqa: ARG002 — kept for API compat
        show_progress_bar: bool = False,  # noqa: ARG002,FBT001,FBT002
    ) -> np.ndarray:
        """Generate embeddings for text(s).

        Args:
            texts: One string or a list of strings.
            batch_size: Ignored — OpenRouter handles batching server-side.
            show_progress_bar: Ignored — no local batching.

        Returns:
            Float32 array of shape (N, EMBED_DIM).
        """
        if isinstance(texts, str):
            texts = [texts]
        return embed(texts)


def get_model() -> EmbeddingModel:
    """Get the singleton embedding model."""
    return EmbeddingModel.get_instance()
