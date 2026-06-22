"""graphknows — unified agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue. ``GraphKnowsRuntime`` is the single transport-neutral
composition root for all memory operations::

    from graphknows import GraphKnowsRuntime

    rt = GraphKnowsRuntime()                       # builds from env / .env
    await rt.ingest_memory("some text", session_id="s1")
    hits = await rt.recall_memory("my query", session_id="s1")

The same runtime backs the MCP stdio server and the web Memory Mirror, so every
transport sees identical behaviour.
"""

from __future__ import annotations

from graphknows.config import GraphKnowsSettings
from graphknows.memory.query import DETRetriever, RetrievalContext
from graphknows.memory.stores.arcadedb import (
    ArcadeDBClient,
    ArcadeDBLTMStore,
    ArcadeDBSTMStore,
    ArcadeDBVectorStore,
)
from graphknows.memory.stores.embedder import embed, embed_one, get_embedder
from graphknows.runtime import GraphKnowsRuntime

__all__ = [
    "ArcadeDBClient",
    "ArcadeDBLTMStore",
    "ArcadeDBSTMStore",
    "ArcadeDBVectorStore",
    "DETRetriever",
    "GraphKnowsRuntime",
    "GraphKnowsSettings",
    "RetrievalContext",
    "embed",
    "embed_one",
    "get_embedder",
]
