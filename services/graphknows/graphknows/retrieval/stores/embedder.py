"""Shared OpenRouter embedder for graphknows (ingestion + retrieval).

Replaces the former sentence-transformers singleton. Uses
``openai/text-embedding-3-small`` via OpenRouter when ``OPENROUTER_API_KEY``
is set; falls back to a zero-vector stub so callers never crash in CI or
environments without the key.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "openai/text-embedding-3-small"
EMBED_DIM: int = 1536  # text-embedding-3-small output dimension
_OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Embed *texts* via OpenRouter, returning shape (N, EMBED_DIM).

    Falls back to zero vectors when the API key is absent or the call fails,
    so ingestion/retrieval paths degrade gracefully rather than crashing.

    Args:
        texts: Input strings to embed.
        model_name: OpenRouter model identifier.

    Returns:
        Float32 array of shape (N, EMBED_DIM).
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — returning zero embeddings")
        return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        return np.array(vecs, dtype=np.float32)
    except Exception as exc:
        logger.warning("OpenRouter embed failed (%s) — returning zero embeddings", exc)
        return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)


def embed_one(text: str, model_name: str = EMBED_MODEL) -> list[float]:
    """Embed a single string and return a plain float list.

    Args:
        text: Input string.
        model_name: OpenRouter model identifier.

    Returns:
        Embedding as a plain Python list of floats.
    """
    return embed([text], model_name)[0].tolist()


def get_embedder(model_name: str = EMBED_MODEL) -> Any:
    """Return a minimal embedder-like object compatible with legacy callers.

    Legacy code expects ``embedder.encode(texts, normalize_embeddings=True)``.
    This shim wraps :func:`embed` so existing call sites keep working without
    sentence-transformers.

    Args:
        model_name: OpenRouter model identifier (unused beyond forwarding).

    Returns:
        Object with an ``encode`` method.
    """

    class _Shim:
        def encode(
            self,
            texts: list[str] | str,
            normalize_embeddings: bool = False,  # noqa: FBT001,FBT002
            show_progress_bar: bool = False,  # noqa: FBT001,FBT002
            batch_size: int = 32,
        ) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            return embed(texts, model_name)

    return _Shim()
