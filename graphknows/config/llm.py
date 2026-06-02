"""LLM configuration helpers for the graphgen package.

Provides:
- :func:`_extract_secret` — safely extract a secret from a config container.
- :func:`_get_openrouter_api_key` — resolve OpenRouter API key.
- :func:`get_model_name` — look up model name by purpose.
- :func:`get_temperature` — look up LLM temperature setting.
- :func:`normalize_openrouter_model` — normalise a model string for DSPy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MASKED_SECRET = "**********"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "moonshotai/kimi-k2"


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
        return value.get_secret_value()
    if isinstance(value, str) and value == _MASKED_SECRET:
        return os.environ.get(key.upper())
    return str(value)


def _get_openrouter_api_key(config: dict[str, Any] | None = None) -> str:
    """Resolve OpenRouter API key from config or environment."""
    if config:
        infra = config.get("infra", {})
        llm = config.get("llm", {})
        if hasattr(llm, "model_dump"):
            llm = llm.model_dump()
        key = _extract_secret(infra, "openrouter_api_key") or _extract_secret(
            llm, "openrouter_api_key"
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


def get_model_name(config: dict[str, Any], purpose: str | None = None) -> str:
    """Return the configured model name for *purpose* (``extraction`` / ``summarization``)."""
    if not config or "llm" not in config:
        raise ValueError("Configuration missing 'llm' section")

    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()

    if purpose == "extraction":
        return llm_cfg.get("extraction_model") or llm_cfg.get("base_model") or _DEFAULT_MODEL
    if purpose == "summarization":
        return llm_cfg.get("summarization_model") or llm_cfg.get("base_model") or _DEFAULT_MODEL
    if purpose == "synthetic":
        return llm_cfg.get("base_model") or _DEFAULT_MODEL

    return llm_cfg.get("base_model") or _DEFAULT_MODEL


def get_temperature(config: dict[str, Any]) -> float:
    """Return the LLM temperature from *config* (defaults to 0.0)."""
    if not config or "llm" not in config:
        return 0.0
    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()
    return float(llm_cfg.get("temperature", 0.0))


def normalize_openrouter_model(model: str | None) -> str:
    """Return *model* with an ``openrouter/`` prefix (idempotent)."""
    if not model:
        return f"openrouter/{_DEFAULT_MODEL}"
    clean = str(model)
    if clean.startswith("openrouter/"):
        clean = clean[len("openrouter/") :]
    return f"openrouter/{clean}"
