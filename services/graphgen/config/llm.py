"""
Centralized LLM Configuration for Knowledge Graph Pipeline.

Simple, single-source configuration for Groq LLM client.
"""

import logging
import os
from typing import Dict, Any, List
from groq import Groq
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


_MASKED_SECRET = "**********"


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


def get_model_name(config: Dict[str, Any], purpose: str = None) -> str:
    """
    Get configured model name.
    Strictly uses config dictionary.
    """
    if not config or 'llm' not in config:
         raise ValueError("Configuration missing 'llm' section")

    llm_cfg = config['llm']
    # Ensure we are working with a dict (in case it wasn't dumped)
    if hasattr(llm_cfg, 'model_dump'):
        llm_cfg = llm_cfg.model_dump()

    if purpose == 'extraction':
        return llm_cfg.get('extraction_model') or llm_cfg.get('base_model')
    elif purpose == 'summarization':
        return llm_cfg.get('summarization_model') or llm_cfg.get('base_model')
    elif purpose == 'synthetic':
        return llm_cfg.get('base_model')

    # General fallback
    return llm_cfg.get('base_model')


def get_temperature(config: Dict[str, Any]) -> float:
    """
    Get LLM temperature setting.
    """
    if not config or 'llm' not in config:
        return 0.0

    llm_cfg = config['llm']
    if hasattr(llm_cfg, 'model_dump'):
        llm_cfg = llm_cfg.model_dump()
    return float(llm_cfg.get('temperature', 0.0))


def normalize_groq_model(model: str | None) -> str:
    """Normalize Groq model identifiers to exactly one groq/ prefix.

    Groq model IDs may legitimately include vendor namespaces like
    meta-llama/llama-4-scout-17b-16e-instruct, so we must preserve the
    model identifier and only add/remove the groq/ provider prefix.
    """
    if not model:
        return "groq/llama-3.1-8b-instant"

    clean_model = str(model)
    if clean_model.startswith("groq/"):
        clean_model = clean_model[len("groq/"):]
    return f"groq/{clean_model}"


def get_langchain_llm(config: Dict[str, Any], purpose: str = None) -> ChatGroq:
    """
    Get LangChain-compatible Groq LLM for use with LangChain tools.

    Used by summarization and retrieval services.

    Args:
        config: Config dictionary
        purpose: Optional purpose ('extraction', 'summarization')

    Returns:
        ChatGroq instance compatible with LangChain tools
    """
    model = get_model_name(config, purpose=purpose)
    temperature = get_temperature(config)

    infra = config.get('infra', {}) if config else {}
    api_key = _extract_secret(infra, 'groq_api_key')

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in configuration or environment")

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )



