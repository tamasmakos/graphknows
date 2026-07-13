"""retrieval — the read path: collect (channels) -> rank (RRF/rerank) -> context.

``build_retriever(settings, vec, ltm, stm)`` is the one place the memory mode
selects a retrieval strategy: ``llm_assisted`` returns the agentic retriever
(which wraps the deterministic engine with a ReAct loop); ``llm_free`` returns
the deterministic engine driven by the configured channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.retrieval.base import BaseGraphRetriever
from graphknows.retrieval.models.context import ChunkMatch, EntityMatch, RetrievalContext
from graphknows.settings import MemoryMode

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_retriever(
    settings: GraphKnowsSettings,
    *,
    vec: Any,
    ltm: Any,
    stm: Any,
) -> Any:
    """Return the retriever for the active memory mode (connected stores injected)."""
    if settings.mode is MemoryMode.llm_assisted:
        from graphknows.retrieval.agentic import AgenticRetriever

        return AgenticRetriever(settings=settings, vec=vec, ltm=ltm, stm=stm)

    from graphknows.channels.registry import default_channels
    from graphknows.retrieval.retriever import DETRetriever

    return DETRetriever(
        vec=vec,
        ltm=ltm,
        stm=stm,
        embed_model=settings.embed_model,
        channels=default_channels(settings),
    )


def __getattr__(name: str) -> Any:
    """Lazily expose the retriever classes without importing the query stack at
    ``import graphknows.retrieval`` time."""
    lazy = {
        "DETRetriever": ("graphknows.retrieval.retriever", "DETRetriever"),
        "AgenticRetriever": ("graphknows.retrieval.agentic", "AgenticRetriever"),
    }
    target = lazy.get(name)
    if target is None:
        raise AttributeError(f"module 'graphknows.retrieval' has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


__all__ = [
    "BaseGraphRetriever",
    "ChunkMatch",
    "EntityMatch",
    "RetrievalContext",
    "build_retriever",
]
