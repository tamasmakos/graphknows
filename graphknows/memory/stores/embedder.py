"""Shared embedder for graphknows (ingestion + retrieval).

Local by default: a `sentence-transformers` model runs in-process, so the
llm_free path needs **no API key**. Set ``GRAPHKNOWS_EMBED_API_BASE`` to route
embeddings through a remote OpenAI-compatible endpoint instead (model + key from
``GRAPHKNOWS_EMBED_MODEL`` / ``GRAPHKNOWS_EMBED_API_KEY``).

No embedding fallbacks: network/model errors propagate directly.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import httpx
import numpy as np

log = logging.getLogger(__name__)

# Default output dimension (BAAI/bge-small-en-v1.5). Updated when a local model
# loads; only used to shape the empty-input result.
EMBED_DIM: int = 384

_MAX_CHARS_PER_TEXT = 20_000  # ~5000 tokens; headroom below typical 8192-token limits


@functools.lru_cache(maxsize=1)
def _settings() -> Any:
    from graphknows.config import GraphKnowsSettings

    return GraphKnowsSettings()


@functools.lru_cache(maxsize=2)
def _local_model(model_name: str) -> Any:
    """Load and cache a local sentence-transformers model on CPU."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("Local embeddings", "local-embeddings") from exc
    model = SentenceTransformer(model_name, device="cpu")
    global EMBED_DIM
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    EMBED_DIM = int(get_dim())
    return model


def _embed_local(texts: list[str], model_name: str) -> np.ndarray:
    model = _local_model(model_name)
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(arr, dtype=np.float32)


def _embed_remote(texts: list[str], model_name: str, api_base: str, api_key: str) -> np.ndarray:
    truncated = [t[:_MAX_CHARS_PER_TEXT] for t in texts]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = httpx.post(
        f"{api_base.rstrip('/')}/embeddings",
        headers=headers,
        json={"model": model_name, "input": truncated},
        timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
    )
    resp.raise_for_status()
    body = resp.json()
    if "data" not in body:
        raise RuntimeError(
            f"Embeddings response missing 'data' key. "
            f"Status {resp.status_code}. Body: {resp.text[:400]}"
        )
    vecs = [d["embedding"] for d in sorted(body["data"], key=lambda x: x["index"])]
    return np.array(vecs, dtype=np.float32)


def embed(texts: list[str], model_name: str = "") -> np.ndarray:
    """Embed *texts*, returning shape ``(N, dim)``.

    Uses a local sentence-transformers model unless ``GRAPHKNOWS_EMBED_API_BASE``
    is set, in which case a remote OpenAI-compatible endpoint is called.

    Args:
        texts: Input strings to embed.
        model_name: Model id override; defaults to the configured ``embed_model``.

    Returns:
        Float32 array of shape ``(N, dim)``.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    s = _settings()
    model = model_name or s.embed_model
    if s.embed_api_base:
        return _embed_remote(texts, model, s.embed_api_base, s.embed_api_key)
    return _embed_local(texts, model)


@functools.lru_cache(maxsize=1)
def embed_dim() -> int:
    """Return the dimension of the configured embedding model (probes once).

    Used to declare vector indexes at the matching dimension.
    """
    return int(embed(["dimension probe"]).shape[1])


@functools.lru_cache(maxsize=4096)
def _embed_one_cached(text: str, model_name: str) -> tuple[float, ...]:
    """LRU-cached single-text embedding keyed by (text, model_name)."""
    return tuple(embed([text], model_name)[0].tolist())


def embed_one(text: str, model_name: str = "") -> list[float]:
    """Embed a single string (LRU-cached) and return a plain float list."""
    return list(_embed_one_cached(text, model_name))


def get_embedder(model_name: str = "") -> Any:
    """Return a minimal ``encode``-compatible embedder shim for legacy callers."""

    class _Shim:
        def encode(self, texts: list[str] | str) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            return embed(texts, model_name)

    return _Shim()
