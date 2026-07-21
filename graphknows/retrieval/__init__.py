"""retrieval — the read path: collect (channels) -> rank (RRF/rerank) -> context.

``build_retriever(settings, store)`` is the one place a retrieval strategy is
selected: ``agentic`` returns the ReAct retriever (which wraps the deterministic
engine), ``det`` returns the deterministic engine driven by the configured
channels. Both read from the one per-namespace
:class:`~graphknows.storage.arcadedb.graph_store.GraphStore`.

The strategy is independent of how the graph was ingested — nothing writes a
mode marker to the store — so it can be chosen per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.retrieval.base import BaseGraphRetriever
from graphknows.retrieval.models.context import ChunkMatch, EntityMatch, RetrievalContext
from graphknows.settings import RetrievalStrategy

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_retriever(
    settings: GraphKnowsSettings,
    *,
    store: Any,
    strategy: str = "",
) -> Any:
    """Return a retriever (connected store injected).

    Args:
        settings: Runtime configuration.
        store: A connected per-namespace GraphStore, shared across strategies.
        strategy: ``det`` or ``agentic``; empty selects
            :attr:`~graphknows.settings.GraphKnowsSettings.retrieval_strategy`.

    Raises:
        ValueError: When ``strategy`` is not a known strategy name.
    """
    resolved = (
        RetrievalStrategy(strategy) if strategy else settings.retrieval_strategy
    )
    if resolved is RetrievalStrategy.agentic:
        from graphknows.retrieval.agentic import AgenticRetriever

        return AgenticRetriever(settings=settings, store=store)

    from graphknows.channels.registry import default_channels
    from graphknows.retrieval.retriever import DETRetriever

    return DETRetriever(
        store=store,
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
