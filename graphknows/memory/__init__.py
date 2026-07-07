# Memory package — STM ingestion, dialog writing, and LTM flush.
#
# Lightweight model re-exports are eager; the heavy re-exports (retriever, the
# ArcadeDB stores, the embedder) are lazy via __getattr__ so ``import graphknows``
# does not pull the query/agentic/dspy or ML-embedding stacks until they are used.

from __future__ import annotations

from typing import Any

from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.models.ingest_options import IngestOptions
from graphknows.memory.models.memory_class import MemoryClass
from graphknows.memory.models.message import Message, MessageInput, Role, normalize_messages
from graphknows.memory.models.scope import MemoryScope

__all__ = [
    "ArcadeDBClient",
    "ArcadeDBLTMStore",
    "ArcadeDBSTMStore",
    "ArcadeDBVectorStore",
    "DETRetriever",
    "IngestOptions",
    "IngestResult",
    "MemoryClass",
    "MemoryScope",
    "Message",
    "MessageInput",
    "RetrievalContext",
    "Role",
    "embed",
    "embed_one",
    "get_embedder",
    "normalize_messages",
]

# name -> (module, attribute) for lazily-loaded heavy re-exports.
_LAZY = {
    "DETRetriever": ("graphknows.memory.query", "DETRetriever"),
    "RetrievalContext": ("graphknows.memory.query", "RetrievalContext"),
    "ArcadeDBClient": ("graphknows.memory.stores.arcadedb", "ArcadeDBClient"),
    "ArcadeDBLTMStore": ("graphknows.memory.stores.arcadedb", "ArcadeDBLTMStore"),
    "ArcadeDBSTMStore": ("graphknows.memory.stores.arcadedb", "ArcadeDBSTMStore"),
    "ArcadeDBVectorStore": ("graphknows.memory.stores.arcadedb", "ArcadeDBVectorStore"),
    "embed": ("graphknows.memory.stores.embedder", "embed"),
    "embed_one": ("graphknows.memory.stores.embedder", "embed_one"),
    "get_embedder": ("graphknows.memory.stores.embedder", "get_embedder"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'graphknows.memory' has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)
