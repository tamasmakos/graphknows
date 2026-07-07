"""graphknows.ports — port interfaces (Protocols) for pluggable infrastructure.

Dependency-free by design: domain code depends on these abstractions, and the
concrete adapters in ``graphknows.config.adapters`` / ``graphknows.memory.stores``
implement them.

**Implementing a backend.** graphknows ships one storage backend (ArcadeDB),
but the store contract is three focused protocols — :class:`ISTMStore`
(session-scoped evidence graph), :class:`IVectorStore` (embeddings + STM/LTM
vector search), and :class:`ILTMStore` (consolidated corpus graph). Implement
all three for your datastore and hand the instances to the runtime via
``Memory(stores=StoreBundle(stm=..., vec=..., ltm=...))``. Extraction and
embedding are similarly swappable via :class:`IEmbedder` and
:class:`IRelationExtractor`.
"""

from __future__ import annotations

from graphknows.ports.embedder import IEmbedder
from graphknows.ports.ltm_store import ILTMStore
from graphknows.ports.relation_extractor import IRelationExtractor
from graphknows.ports.stm_store import ISTMStore
from graphknows.ports.vector_store import IVectorStore

# Deprecated alias — removed in 2.0. Prefer IVectorStore.
IVectorDb = IVectorStore

__all__ = [
    "IEmbedder",
    "ILTMStore",
    "IRelationExtractor",
    "ISTMStore",
    "IVectorDb",
    "IVectorStore",
]
