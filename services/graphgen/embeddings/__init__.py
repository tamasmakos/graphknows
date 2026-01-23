"""
Embedding generation modules.

Provides:
- RAG embeddings using SentenceTransformers
- KGE embeddings using PyKEEN (optional)
"""

from .rag import (
    generate_rag_embeddings,
    get_embedding_dimension,
)

__all__ = [
    'generate_rag_embeddings',
    'get_embedding_dimension',
]
