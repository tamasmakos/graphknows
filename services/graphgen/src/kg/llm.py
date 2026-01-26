"""
Centralized LLM Configuration for Knowledge Graph Pipeline.

Simple, single-source configuration for Groq LLM client.
"""

import logging
from typing import Dict, Any, List
from groq import Groq
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


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
    
    api_key = None
    if config and 'infra' in config:
        infra = config['infra']
        # Handle Pydantic object or dict
        if hasattr(infra, 'model_dump'):
            infra = infra.model_dump()
            
        api_key_val = infra.get('groq_api_key')
        if api_key_val:
            # Handle SecretStr if it hasn't been effectively serialized to str yet
            if hasattr(api_key_val, 'get_secret_value'):
                api_key = api_key_val.get_secret_value()
            else:
                api_key = str(api_key_val)

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in configuration")
    
    return ChatGroq(
        model=model,
        temperature=temperature,
    )



