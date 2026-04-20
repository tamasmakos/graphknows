"""
LlamaIndex AgentWorkflow — iterative graph RAG agent.

Uses 4 Neo4j retrieval tools in an FunctionCallingAgent loop.
Returns a structured AgentResponse with citations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


def _make_llm(settings: Any):
    """Instantiate the LLM from the configured provider."""
    from llama_index.core.llms import LLM  # type: ignore[import]

    provider = (settings.llm_provider or "groq").lower()
    model = settings.graphrag_chat_model or settings.llm_model

    if provider == "groq":
        from llama_index.llms.groq import Groq  # type: ignore[import]

        return Groq(model=model, api_key=settings.llm_groq_api_key)
    elif provider == "openai":
        from llama_index.llms.openai import OpenAI  # type: ignore[import]

        return OpenAI(model=model, api_key=settings.llm_openai_api_key)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}")


def _build_tools(driver: AsyncDriver, database: str) -> list:
    """Wrap retrieval functions as LlamaIndex FunctionTool objects."""
    from llama_index.core.tools import FunctionTool  # type: ignore[import]
    from src.agent.tools import (
        search_chunks,
        get_entity_neighbours,
        get_document_context,
        search_entities,
    )
    import asyncio
    import functools

    def _sync(coro_fn, **fixed):
        """Make an async tool function synchronous (LlamaIndex FunctionTool is sync)."""

        @functools.wraps(coro_fn)
        def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro_fn(*args, **fixed, **kwargs))

        return wrapper

    return [
        FunctionTool.from_defaults(
            fn=_sync(search_chunks, driver=driver, database=database),
            name="search_chunks",
            description=(
                "Search document chunks by semantic similarity. "
                "Use this first for any factual question. "
                "Args: query (str), k (int, default 5)."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_sync(get_entity_neighbours, driver=driver, database=database),
            name="get_entity_neighbours",
            description=(
                "Get entities related to a given entity by name. "
                "Use to explore relationships around a known entity. "
                "Args: entity_name (str), depth (int, default 1)."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_sync(get_document_context, driver=driver, database=database),
            name="get_document_context",
            description=(
                "Fetch metadata and sample content for a specific document by doc_id. "
                "Args: doc_id (str)."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_sync(search_entities, driver=driver, database=database),
            name="search_entities",
            description=(
                "Search entities by semantic similarity to a query. "
                "Use to find relevant people, organisations, concepts, or locations. "
                "Args: query (str), k (int, default 10)."
            ),
        ),
    ]


async def run_agent(
    query: str,
    driver: AsyncDriver,
    database: str = "neo4j",
    chat_history: list | None = None,
) -> dict[str, Any]:
    """
    Run the AgentWorkflow for a single query.

    Returns a dict compatible with AgentResponse.
    """
    from llama_index.core.agent import ReActAgent  # type: ignore[import]
    from src.common.config.settings import AppSettings

    settings = AppSettings()
    llm = _make_llm(settings)
    tools = _build_tools(driver, database)

    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        max_iterations=10,
        context=(
            "You are a knowledge-graph RAG assistant. "
            "Always use the provided tools to look up information before answering. "
            "Cite your sources using [n] markers linked to the chunks you retrieved."
        ),
    )

    start = time.time()
    response = await agent.achat(query, chat_history=chat_history or [])
    elapsed = time.time() - start

    answer_text = str(response)

    # Extract citations from tool call results stored in agent memory
    citations = _extract_citations(agent)

    return {
        "answer": answer_text,
        "citations": citations,
        "graph_data": {"nodes": [], "edges": []},
        "execution_time": elapsed,
    }


async def stream_agent(
    query: str,
    driver: AsyncDriver,
    database: str = "neo4j",
    chat_history: list | None = None,
) -> AsyncIterator[str]:
    """
    Stream agent responses as SSE-compatible JSON strings.

    Each yielded item is a JSON string with {"type": ..., "data": ...}.
    """
    from llama_index.core.agent import ReActAgent  # type: ignore[import]
    from src.common.config.settings import AppSettings

    settings = AppSettings()
    llm = _make_llm(settings)
    tools = _build_tools(driver, database)

    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=False,
        max_iterations=10,
        context=(
            "You are a knowledge-graph RAG assistant. "
            "Always use the provided tools to look up information before answering. "
            "Cite your sources using [n] markers linked to the chunks you retrieved."
        ),
    )

    # Stream token-by-token
    streaming_response = await agent.astream_chat(query, chat_history=chat_history or [])

    async for delta in streaming_response.async_response_gen():
        yield json.dumps({"type": "delta", "data": delta})

    # Emit citations after streaming completes
    citations = _extract_citations(agent)
    yield json.dumps({"type": "citations", "data": citations})
    yield json.dumps({"type": "done", "data": None})


def _extract_citations(agent) -> list[dict[str, Any]]:
    """
    Extract citation objects from the agent's tool call memory.
    Looks for search_chunks results and maps them to Citation schema.
    """
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    try:
        for source_node in getattr(agent, "source_nodes", []):
            cid = getattr(source_node, "node_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            meta = getattr(source_node, "metadata", {}) or {}
            citations.append(
                {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "heading_path": meta.get("heading_path", []),
                    "text_excerpt": (source_node.get_text() or "")[:300],
                    "score": getattr(source_node, "score", 0.0) or 0.0,
                }
            )
    except Exception as exc:
        logger.debug("Citation extraction failed: %s", exc)

    return citations
