"""
LlamaIndex embeddings adapter using HuggingFace SentenceTransformers.

Wraps the existing SentenceTransformer model to be compatible with LlamaIndex.
"""

from typing import List

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


_embedding_model = None


def get_llamaindex_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbedding:
    """
    Get a LlamaIndex-compatible HuggingFace embedding model.

    Uses the same model as the existing pipeline for consistency.

    Args:
        model_name: HuggingFace model name for embeddings

    Returns:
        Configured HuggingFaceEmbedding instance
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            # Try loading from local cache first
            _embedding_model = HuggingFaceEmbedding(model_name=model_name, local_files_only=True)
        except Exception:
            # Fallback to default loading
            _embedding_model = HuggingFaceEmbedding(model_name=model_name)
    return _embedding_model


def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Embed a single query text.

    Convenience function for direct embedding without managing model lifecycle.

    Args:
        text: Text to embed
        model_name: HuggingFace model name

    Returns:
        Embedding vector as list of floats
    """
    model = get_llamaindex_embeddings(model_name)
    return model.get_query_embedding(text)
