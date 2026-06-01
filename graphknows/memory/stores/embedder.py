"""Shared OpenRouter embedder for graphknows (ingestion + retrieval).

Uses ``openai/text-embedding-3-small`` via OpenRouter. Raises ``RuntimeError``
when ``OPENROUTER_API_KEY`` is not set, and propagates HTTP/network exceptions
directly — no zero-vector fallbacks, no silent degradation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "openai/text-embedding-3-small"
EMBED_DIM: int = 1536  # text-embedding-3-small output dimension
_OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


_MAX_CHARS_PER_TEXT = 20_000  # ~5000 tokens; leave headroom below 8192-token limit


def _clean_secret(value: str | None) -> str:
    """Return a usable secret from shell or .env syntax."""
    return (value or "").strip().strip('"').strip("'")


def _repo_env_key() -> str:
    """Best-effort OpenRouter key lookup from the repo .env file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            values = dotenv_values(candidate)
            return _clean_secret(
                values.get("OPENROUTER_API_KEY") or values.get("OPEN_ROUTER_API")
            )
    return ""


def _openrouter_api_key() -> str:
    """Resolve and normalize the OpenRouter key for all embedder call paths."""
    key = _clean_secret(os.environ.get("OPENROUTER_API_KEY")) or _clean_secret(
        os.environ.get("OPEN_ROUTER_API")
    )
    if not key:
        key = _repo_env_key()
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        os.environ.setdefault("OPEN_ROUTER_API", key)
    return key


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Embed *texts* via OpenRouter, returning shape (N, EMBED_DIM).

    Each text is silently truncated to ``_MAX_CHARS_PER_TEXT`` characters before
    sending to avoid OpenRouter's per-text token limit.

    Args:
        texts: Input strings to embed.
        model_name: OpenRouter model identifier.

    Returns:
        Float32 array of shape (N, EMBED_DIM).

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is not set.
        httpx.HTTPStatusError: On HTTP errors from the OpenRouter API.
        httpx.RequestError: On network-level failures.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    api_key = _openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — embedding requires the OpenRouter API key. "
            "Set it in .env or the environment before running."
        )

    truncated = [t[:_MAX_CHARS_PER_TEXT] for t in texts]
    resp = httpx.post(
        _OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_name, "input": truncated},
        timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
    )
    resp.raise_for_status()
    body = resp.json()
    if "data" not in body:
        raise RuntimeError(
            f"OpenRouter embeddings response missing 'data' key. "
            f"Status {resp.status_code}. Body: {resp.text[:400]}"
        )
    data = body["data"]
    vecs = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
    return np.array(vecs, dtype=np.float32)


def embed_one(text: str, model_name: str = EMBED_MODEL) -> list[float]:
    """Embed a single string and return a plain float list.

    Args:
        text: Input string.
        model_name: OpenRouter model identifier.

    Returns:
        Embedding as a plain Python list of floats.

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is not set.
        httpx.HTTPStatusError: On HTTP errors from the OpenRouter API.
    """
    return embed([text], model_name)[0].tolist()


def get_embedder(model_name: str = EMBED_MODEL) -> Any:
    """Return a minimal embedder-like object compatible with legacy callers.

    Legacy code expects ``embedder.encode(texts, normalize_embeddings=True)``.
    This shim wraps :func:`embed` so existing call sites keep working.

    Args:
        model_name: OpenRouter model identifier.

    Returns:
        Object with an ``encode`` method.
    """

    class _Shim:
        def encode(
            self,
            texts: list[str] | str,
            batch_size: int = 32,
        ) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            return embed(texts, model_name)

    return _Shim()
