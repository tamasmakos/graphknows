"""Lazy settings + store accessors for MCP tool modules."""

from __future__ import annotations

from typing import Any

from graphknows.config import GraphKnowsSettings

_settings: GraphKnowsSettings | None = None
_vector: Any = None
_retriever: Any = None
_runtime: Any = None


def get_settings() -> GraphKnowsSettings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def get_runtime() -> Any:
    """Return the singleton transport-neutral graphknows runtime."""
    global _runtime
    if _runtime is None:
        from graphknows.runtime import GraphKnowsRuntime

        _runtime = GraphKnowsRuntime(get_settings())
    return _runtime


def get_stm_store() -> Any:
    """Build an ArcadeDBSTMStore from current settings (not yet connected)."""
    from graphknows.config.builders.stores import build_arcadedb_stm_store

    return build_arcadedb_stm_store(get_settings())


async def get_vector_store() -> Any:
    """Return a connected ArcadeDBVectorStore singleton."""
    global _vector
    if _vector is None:
        from graphknows.config.builders.stores import build_arcadedb_vector_store

        _vector = build_arcadedb_vector_store(get_settings())
        await _vector.connect()
        await _vector.ensure_schema()
    return _vector


async def get_ltm_store() -> Any:
    """Return a connected ArcadeDBLTMStore."""
    from graphknows.config.builders.stores import build_arcadedb_ltm_store

    store = build_arcadedb_ltm_store(get_settings())
    await store._c.connect()
    await store.ensure_schema()
    return store


async def get_or_init_retriever() -> Any:
    """Return the singleton retriever, initialising it on first call."""
    global _retriever
    if _retriever is not None:
        return _retriever
    s = get_settings()
    vector = await get_vector_store()
    ltm = await get_ltm_store()
    stm = get_stm_store()
    await stm._c.connect()
    if s.retrieval_mode == "agentic":
        from graphknows.memory.query.agentic import AgenticRetriever

        _retriever = AgenticRetriever(
            settings=s,
            pg=vector,
            neo=ltm,
            stm=stm,
        )
    else:
        from graphknows.memory.query.retriever import DETRetriever

        _retriever = DETRetriever(pg=vector, neo=ltm, mg=stm, embed_model=s.embed_model)
    return _retriever
