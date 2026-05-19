"""Shared BGE sentence-transformer embedder singleton for graphknows.

Single source of truth for both ingestion (graphknows/ingestion) and
retrieval (graphknows/retrieval) so both use the same embedding space.
"""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384

_model: SentenceTransformer | None = None


def get_embedder(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """Return the singleton sentence-transformer, loading it on first call."""
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Embed *texts* with L2-normalised BGE-small, returning shape (N, 384)."""
    return get_embedder(model_name).encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_one(text: str, model_name: str = EMBED_MODEL) -> list[float]:
    """Embed a single string and return a plain float list."""
    return embed([text], model_name)[0].tolist()
