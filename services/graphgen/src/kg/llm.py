"""
Centralized LLM Configuration for Knowledge Graph Pipeline.

Simple, single-source configuration for Groq LLM client.
"""

import os
import logging
from typing import Dict, Any, List
from groq import Groq
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def get_model_name(config: Dict[str, Any] = None, purpose: str = None) -> str:
    """
    Get configured model name.
    
    Priority:
    1. Purpose-specific Env Var (EXTRACTION_MODEL, SUMMARISATION_MODEL)
    2. Config dict 'llm_model' key
    3. GROQ_MODEL environment variable
    4. Hardcoded Defaults
    """
    # 0. Check purpose-specific env vars
    if purpose == 'extraction':
        return os.environ.get("EXTRACTION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    elif purpose == 'summarization':
        return os.environ.get("SUMMARISATION_MODEL", "meta-llama/llama-guard-4-12b")
    elif purpose == 'synthetic':
        return os.environ.get("SYNTH_MODEL", "llama-3.3-70b-versatile")

    if config:
        if 'llm_model' in config:
            return config['llm_model']
        if 'llm' in config and isinstance(config['llm'], dict) and 'model' in config['llm']:
            return config['llm']['model']
    
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_temperature(config: Dict[str, Any] = None) -> float:
    """
    Get LLM temperature setting.
    
    Args:
        config: Optional config dictionary
        
    Returns:
        Temperature value (0.0 to 1.0)
    """
    if config and 'llm_temperature' in config:
        return float(config['llm_temperature'])
    
    return float(os.environ.get("LLM_TEMPERATURE", "0.0"))








def get_langchain_llm(config: Dict[str, Any] = None, purpose: str = None) -> ChatGroq:
    """
    Get LangChain-compatible Groq LLM for use with LangChain tools.
    
    Used by summarization and retrieval services.
    
    Args:
        config: Optional config dictionary
        purpose: Optional purpose ('extraction', 'summarization')
        
    Returns:
        ChatGroq instance compatible with LangChain tools
    """
    model = get_model_name(config, purpose=purpose)
    temperature = get_temperature(config)
    api_key = os.environ.get("GROQ_API_KEY")
    
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )



