"""LLM configuration helpers for graphknows.

Provides:
- :func:`build_lm` — construct a provider-neutral ``dspy.LM``.
- :func:`_extract_secret` — safely extract a secret from a config container.
- :func:`get_openrouter_api_key` — resolve OpenRouter API key.
- :func:`normalize_openrouter_model` — normalise a model string for DSPy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# litellm fetches its model-cost map from GitHub when it is imported, unless
# told to use the copy it ships. The only egress an LLM-backed capability is
# allowed is the configured provider, and this module is the one every dspy
# user in the package imports before dspy (the decoder, the topic summariser),
# so the switch is set here. setdefault: an operator who set it keeps their
# value.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_MASKED_SECRET = "**********"
_DEFAULT_MODEL = ""


def build_lm(
    model: str,
    *,
    api_key: str = "",
    api_base: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    **extra: Any,
) -> Any:
    """Construct a provider-neutral ``dspy.LM``.

    ``model`` is a litellm model string (e.g. ``openrouter/deepseek/deepseek-v4-flash``
    or ``openai/gpt-4o-mini``); the provider base URL is inferred from its prefix.
    Pass ``api_base`` only to override that (e.g. a self-hosted vLLM endpoint). This
    is the single place graphknows builds an LM — no provider endpoint is hardcoded.
    """
    import dspy

    # OpenRouter model ids carry a nested provider (e.g.
    # openrouter/deepseek/deepseek-v4-flash). Without an explicit base, litellm
    # can strip the openrouter/ prefix and mis-route the nested provider to its
    # native API. Pin the OpenRouter base so the default config works out of the
    # box; any other provider is left to litellm's own inference.
    if not api_base and model.startswith("openrouter/"):
        api_base = "https://openrouter.ai/api/v1"

    # litellm sets no socket timeout of its own, so a dropped upstream
    # connection neither raises nor returns — the await simply never completes
    # and the whole run wedges with the process alive and its log frozen
    # (observed: 2h49m on a single judge call). Every LM this repo builds gets a
    # ceiling; pass timeout=... to widen it for a deliberately long call.
    # dspy caches every prompt and completion under ~/.dspy_cache by default,
    # which puts ingested chunk text on disk outside the configured database.
    # Off for every LM this repo builds; pass cache=True to opt one back in.
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "cache": False,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    kwargs.update(extra)
    return dspy.LM(**kwargs)


def _as_mapping(cfg: Any) -> dict[str, Any]:
    """Return *cfg* as a plain mapping, unwrapping a pydantic model if needed.

    Callers hand these helpers a dict, a pydantic settings object, or nothing;
    each accessor used to re-implement the same unwrap.
    """
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()  # type: ignore[no-any-return]
    return cfg if isinstance(cfg, dict) else {}


def _extract_secret(container: Any, key: str) -> str | None:
    """Safely extract a secret from dicts, pydantic models, or env-backed config."""
    if container is None:
        return None

    if hasattr(container, key):
        value = getattr(container, key)
    elif isinstance(container, dict):
        value = container.get(key)
    else:
        value = None

    if not value:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()  # type: ignore[no-any-return]
    if isinstance(value, str) and value == _MASKED_SECRET:
        return os.environ.get(key.upper())
    return str(value)  # type: ignore[return-value]


def get_openrouter_api_key(config: dict[str, Any] | None = None) -> str:
    """Resolve OpenRouter API key from config or environment."""
    if config:
        infra = config.get("infra", {})
        llm = _as_mapping(config.get("llm", {}))
        key = (
            _extract_secret(config, "openrouter_api_key")
            or _extract_secret(infra, "openrouter_api_key")
            or _extract_secret(llm, "openrouter_api_key")
        )
        if key:
            return key

    def _resolve(val: str | None) -> str:
        return val if (val and val != _MASKED_SECRET) else ""

    return (
        _resolve(os.environ.get("OPENROUTER_API_KEY"))
        or _resolve(os.environ.get("OPEN_ROUTER_API"))
        or ""
    )


def normalize_openrouter_model(model: str | None) -> str:
    """Return *model* with an ``openrouter/`` prefix (idempotent)."""
    if not model:
        return f"openrouter/{_DEFAULT_MODEL}"
    clean = str(model)
    if clean.startswith("openrouter/"):
        clean = clean[len("openrouter/") :]
    return f"openrouter/{clean}"
