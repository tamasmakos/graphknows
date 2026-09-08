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

from graphknows.exceptions import ConfigurationError

log = logging.getLogger(__name__)

# Default output dimension, matching settings.embed_model (BAAI/bge-small-en-v1.5).
# Updated when a local model loads; only used to shape the empty-input result.
EMBED_DIM: int = 384

_MAX_CHARS_PER_TEXT = 20_000  # ~5000 tokens; headroom below typical 8192-token limits


def _settings() -> Any:
    from graphknows.settings import get_settings

    return get_settings()


@functools.lru_cache(maxsize=2)
def _local_model(model_name: str) -> Any:
    """Load and cache a local sentence-transformers model.

    Device and quantization come from settings (``embed_device`` /
    ``embed_quantization``). bitsandbytes 4-/8-bit quantization applies only on
    CUDA; on CPU the model loads at full bf16 precision.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        # sentence-transformers is a BASE dependency, not an extra: the batteries
        # -included install promises working local embeddings. Its absence means a
        # broken environment, so point at that rather than at an extra to install
        # (there is no 'local-embeddings' extra, and suggesting one sends people
        # down a dead end).
        from graphknows.exceptions import BrokenInstallError

        raise BrokenInstallError(
            "Local embeddings",
            "sentence-transformers",
            hint="To use a remote embedding endpoint instead, set GRAPHKNOWS_EMBED_API_BASE.",
        ) from exc

    import torch

    s = _settings()
    want = (getattr(s, "embed_device", "auto") or "auto").lower()
    device = "cuda" if want in ("auto", "cuda") and torch.cuda.is_available() else "cpu"

    # zembed-1 ships custom modelling code and is trained in bf16.
    model_kwargs: dict[str, Any] = {"torch_dtype": "bfloat16"}
    quant = (getattr(s, "embed_quantization", "") or "").lower()
    if quant in ("4bit", "8bit") and device == "cuda":
        # transformers ships no stubs for BitsAndBytesConfig, so every call is
        # `no-untyped-call` — a third-party typing gap, not a defect here.
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]
        # A quantized model is placed on-device by accelerate; passing device= would
        # make sentence-transformers call .to(cuda), which quantized models reject.
        model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
    else:
        model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True, model_kwargs=model_kwargs
        )

    global EMBED_DIM
    # Both spellings are typed `int | None`; int(None) would surface a module stack
    # that exposes no dimension as a TypeError naming neither the model nor the cause.
    get_dim = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    dim = get_dim()
    if dim is None:
        raise ConfigurationError(f"Embedding model {model_name!r} reports no dimension.")
    EMBED_DIM = int(dim)
    return model


@functools.lru_cache(maxsize=8)
def _model_suffix(model_name: str) -> str:
    r"""The ``suffix`` a model declares in ``config_sentence_transformers.json``.

    sentence-transformers applies ``prompts`` — which are **prefixes only** — and
    has no suffix support (5.6.1: no ``suffix`` attribute, no mention in
    ``encode``). A model that declares one therefore never gets it.

    That is harmless for mean-pooled models and severe for last-token pooling:
    the pooled vector is read at the final token, so without the declared
    sentinel the model pools at whatever content token happens to end the text.
    zembed-1 (``pooling_mode_lasttoken``, ``suffix: "<|im_end|>\n"``) inverted
    ontology class matching outright — ``Possession`` ranked 11/11 for a sentence
    about buying a motorcycle, behind ``Emotion``.

    Returns "" for the common case (bge-small declares no suffix) and for any
    lookup failure — embedding is on the ingest hot path and must not raise here.
    """
    import json
    from pathlib import Path

    config = Path(model_name) / "config_sentence_transformers.json"
    if not config.is_file():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config_sentence_transformers.json")
        except Exception:  # offline, bad repo id, hub API change
            return ""
        if not isinstance(cached, str):
            return ""
        config = Path(cached)
    try:
        declared = json.loads(config.read_text(encoding="utf-8")).get("suffix")
    except (OSError, ValueError):
        return ""
    return str(declared) if declared else ""


model_suffix = _model_suffix


def _embed_local(texts: list[str], model_name: str) -> np.ndarray:
    model = _local_model(model_name)
    # Applied here rather than at the call sites so every embedding — ingest,
    # query, ontology gloss — is encoded the way the model expects.
    suffix = _model_suffix(model_name)
    if suffix:
        texts = [t + suffix for t in texts]
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(arr, dtype=np.float32)


def _embed_remote(
    texts: list[str], model_name: str, api_base: str, api_key: str, dimensions: int = 0
) -> np.ndarray:
    truncated = [t[:_MAX_CHARS_PER_TEXT] for t in texts]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {"model": model_name, "input": truncated}
    if dimensions > 0:
        # Matryoshka truncation server-side; the API re-normalizes the vectors.
        body["dimensions"] = dimensions
    resp = httpx.post(
        f"{api_base.rstrip('/')}/embeddings",
        headers=headers,
        json=body,
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
        return _embed_remote(
            texts,
            model,
            s.embed_api_base,
            s.embed_api_key.get_secret_value(),
            dimensions=getattr(s, "embed_dimensions", 0),
        )
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
