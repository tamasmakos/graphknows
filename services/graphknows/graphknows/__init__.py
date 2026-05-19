"""graphknows — unified agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue::

    from graphknows import GraphKnows

    async with GraphKnows.from_env() as gk:
        # Framework-agnostic tool list (name + description + JSON Schema + async callable)
        tools = gk.tools          # list[GraphKnowsTool]

        # MCP server — for out-of-process / MCP-native agents
        mcp_srv = gk.as_mcp_server()

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

from graphknows.config import GraphKnowsSettings
from graphknows.container import GraphKnowsContainer
from graphknows.graphgen import run_ingestion
from graphknows.models.tools import GraphKnowsTool
from graphknows.retrieval import HybridRetriever, RetrievalContext
from graphknows.retrieval.stores import (
    MemgraphStore,
    Neo4jStore,
    PgVectorStore,
    embed,
    embed_one,
    get_embedder,
)

# Ergonomic top-level alias — the most common import pattern:
#   from graphknows import GraphKnows
GraphKnows = GraphKnowsContainer

__all__ = [
    # Primary entry-point
    "GraphKnows",
    "GraphKnowsContainer",
    # Configuration
    "GraphKnowsSettings",
    # Tool primitive
    "GraphKnowsTool",
    # Retrieval
    "HybridRetriever",
    "RetrievalContext",
    # Stores
    "Neo4jStore",
    "PgVectorStore",
    "MemgraphStore",
    # Embedder
    "get_embedder",
    "embed",
    "embed_one",
    # Ingestion
    "run_ingestion",
]
