"""
LlamaIndex embeddings adapter.
"""
import logging
import os

logger = logging.getLogger(__name__)

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

_instance = None

def get_llamaindex_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Get the singleton HuggingFace embedding model."""
    global _instance
    if _instance:
        return _instance

    if not HuggingFaceEmbedding:
        raise ImportError("llama-index-embeddings-huggingface not installed.")

    # Align with global config if present
    model_name = os.getenv("EMBEDDING_MODEL", model_name)
    
    logger.info(f"Initializing embedding model: {model_name}")
    
    # Simple and robust initialization
    # We rely on sentence-transformers (used by llama-index) to handle caching/downloads
    _instance = HuggingFaceEmbedding(model_name=model_name)
    return _instance

def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2"):
    """Direct embedding query."""
    return get_llamaindex_embeddings(model_name).get_query_embedding(text)