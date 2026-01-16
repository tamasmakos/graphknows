"""
LlamaIndex LLM configuration using Groq.

Provides a configured Groq LLM instance compatible with LlamaIndex workflows.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()


@lru_cache(maxsize=1)
def get_llamaindex_llm() -> Groq:
    """
    Get a LlamaIndex-compatible Groq LLM instance.

    Reads configuration from environment variables:
    - GROQ_API_KEY: API key for Groq
    - GROQ_MODEL: Model name (default: llama-3.3-70b-versatile)

    Returns:
        Configured Groq LLM instance
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    return Groq(
        model=model,
        api_key=api_key,
        temperature=0.1,
    )
