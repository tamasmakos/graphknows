"""
Graph processing modules.

Provides utilities for:
- Graph extraction (entities, relations)
- Centrality calculations
- Coreference resolution
- Embedding similarity computation
"""

from .similarity import (
    compute_embedding_similarity_edges,
)

__all__ = [
    'compute_embedding_similarity_edges',
]
