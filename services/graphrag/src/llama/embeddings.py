"""
LlamaIndex embeddings adapter.

Supports HuggingFace SentenceTransformers (if available) and OpenAI embeddings.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    OpenAIEmbedding = None


_embedding_model = None


def get_llamaindex_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get a LlamaIndex-compatible embedding model.

    Prioritizes HuggingFace (if available and requested), falls back to OpenAI.

    Args:
        model_name: HuggingFace model name for embeddings

    Returns:
        Configured Embedding instance
    """
    global _embedding_model
    if _embedding_model is None:
        if HuggingFaceEmbedding:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading HuggingFace embedding ({model_name}) on {device}...")
                # Try loading from local cache first
                _embedding_model = HuggingFaceEmbedding(model_name=model_name, local_files_only=True, device=device)
            except Exception:
                try:
                    # Fallback to default loading
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    _embedding_model = HuggingFaceEmbedding(model_name=model_name, device=device)
                except Exception as e:
                    logger.warning(f"Failed to load HuggingFace embedding: {e}")
        
        if _embedding_model is None:
            if OpenAIEmbedding:
                logger.info("Using OpenAI embeddings fallback.")
                _embedding_model = OpenAIEmbedding()
            else:
                raise RuntimeError("No embedding model available. Install llama-index-embeddings-huggingface (with torch) or llama-index-embeddings-openai.")
                
    return _embedding_model


def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Embed a single query text.

    Convenience function for direct embedding without managing model lifecycle.

    Args:
        text: Text to embed
        model_name: HuggingFace model name (if used)

    Returns:
        Embedding vector as list of floats
    """
    model = get_llamaindex_embeddings(model_name)
    return model.get_query_embedding(text)