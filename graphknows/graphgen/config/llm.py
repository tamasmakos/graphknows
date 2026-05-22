"""
Centralized LLM Configuration for Knowledge Graph Pipeline.

Uses OpenRouter as the sole LLM provider.
"""

import logging
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI

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


def _get_openrouter_api_key(config: Dict[str, Any] | None = None) -> str:
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

    # Support both env-var spellings present in .env
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API") or ""


def get_model_name(config: Dict[str, Any], purpose: str = None) -> str:
    """Get configured model name from config dictionary."""
    if not config or "llm" not in config:
        raise ValueError("Configuration missing 'llm' section")

    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()

    if purpose == "extraction":
        return llm_cfg.get("extraction_model") or llm_cfg.get("base_model") or _DEFAULT_MODEL
    elif purpose == "summarization":
        return llm_cfg.get("summarization_model") or llm_cfg.get("base_model") or _DEFAULT_MODEL
    elif purpose == "synthetic":
        return llm_cfg.get("base_model") or _DEFAULT_MODEL

    return llm_cfg.get("base_model") or _DEFAULT_MODEL


def get_temperature(config: Dict[str, Any]) -> float:
    """Get LLM temperature setting."""
    if not config or "llm" not in config:
        return 0.0

    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()
    return float(llm_cfg.get("temperature", 0.0))


def normalize_openrouter_model(model: str | None) -> str:
    """Normalize a model name to the openrouter/ DSPy prefix.

    Strips any existing openrouter/ prefix before adding it, so the
    function is idempotent regardless of how the model string arrives.
    """
    if not model:
        return f"openrouter/{_DEFAULT_MODEL}"

    clean = str(model)
    if clean.startswith("openrouter/"):
        clean = clean[len("openrouter/") :]
    return f"openrouter/{clean}"


def get_langchain_llm(config: Dict[str, Any], purpose: str = None) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI instance pointed at OpenRouter.

    Drop-in replacement for the former ChatGroq helper; callers do not
    need to change — the returned object satisfies the same interface.
    """
    model = get_model_name(config, purpose=purpose)
    temperature = get_temperature(config)
    api_key = _get_openrouter_api_key(config)

    if not api_key:
        raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY or OPEN_ROUTER_API.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
    )
