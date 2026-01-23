"""
LlamaIndex integration package for the KG Agent.

This package provides LlamaIndex-compatible adapters and utilities
for integrating with FalkorDB, Groq LLMs, and HuggingFace embeddings.
"""

from src.llama.llm import get_llamaindex_llm
from src.llama.embeddings import get_llamaindex_embeddings
from src.llama.graph_store import get_graph_store

__all__ = [
    "get_llamaindex_llm",
    "get_llamaindex_embeddings", 
    "get_graph_store",
]
