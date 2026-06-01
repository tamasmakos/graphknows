"""graphknows — unified agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue::

    from graphknows import GraphKnows

    async with GraphKnows.from_env() as gk:
        # Framework-agnostic tool list (name + description + JSON Schema + async callable)
        tools = gk.tools          # list[GraphKnowsTool]

        # Direct programmatic access
        ctx = await gk.retriever.retrieve("my query", session_id="s1")

Wrapping tools for a specific framework takes one line by the caller:

    # LangChain
    lc_tools = [StructuredTool.from_function(t.fn, name=t.name, description=t.description)
                for t in gk.tools]

    # OpenAI Agents SDK
    oa_tools = [FunctionTool(t.fn, name=t.name, description=t.description)
                for t in gk.tools]

    # PydanticAI
    pa_tools = [Tool(t.fn, name=t.name, description=t.description) for t in gk.tools]

No agent framework is imported by this package.
"""

from __future__ import annotations

from graphknows.config import GraphKnowsSettings
from graphknows.config.wiring import build_service
from graphknows.container import GraphKnowsContainer
from graphknows.mcp.tool import GraphKnowsTool
from graphknows.memory.query import HybridRetriever, RetrievalContext
from graphknows.memory.stores.embedder import embed, embed_one, get_embedder
from graphknows.memory.stores.memgraph import MemgraphStore
from graphknows.memory.stores.neo4j import Neo4jStore
from graphknows.memory.stores.pgvector import PgVectorStore

# Ergonomic top-level alias — the most common import pattern:
#   from graphknows import GraphKnows
GraphKnows = GraphKnowsContainer

__all__ = [
    "GraphKnows",
    "GraphKnowsContainer",
    "GraphKnowsSettings",
    "GraphKnowsTool",
    "HybridRetriever",
    "MemgraphStore",
    "Neo4jStore",
    "PgVectorStore",
    "RetrievalContext",
    "build_service",
    "embed",
    "embed_one",
    "get_embedder",
]
