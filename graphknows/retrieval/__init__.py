"""retrieval — the read path: collect (channels) -> rank (RRF) -> context.

``build_retriever(settings, store=...)`` returns the one retriever there is: the
deterministic engine, driven by the fixed ``core_channels()`` plus whatever the
caller passes, reading the one per-namespace
:class:`~graphknows.storage.arcadedb.graph_store.GraphStore`. Nothing writes a
mode marker to the store, so any namespace can be read this way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.retrieval.models.context import RetrievalContext

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_retriever(
    settings: GraphKnowsSettings, *, store: Any, extra_channels: list[Any] | None = None
) -> Any:
    """Return the deterministic retriever (connected store injected).

    Args:
        settings: Runtime configuration.
        store: A connected per-namespace GraphStore.
        extra_channels: Caller-supplied channels, appended after the core
            collectors. This is how an application adds a signal
            over its own graph structure without editing the registry: the same
            objects whose ``populate`` wrote that structure at flush time are
            the ones whose ``collect`` reads it back here.
    """
    from graphknows.channels.registry import core_channels
    from graphknows.retrieval.retriever import DETRetriever

    return DETRetriever(
        store=store,
        embed_model=settings.embed_model,
        channels=[*core_channels(), *(extra_channels or [])],
    )


def __getattr__(name: str) -> Any:
    """Lazily expose the retriever classes.

    Keeps the query stack out of ``import graphknows.retrieval``.
    """
    lazy = {
        "DETRetriever": ("graphknows.retrieval.retriever", "DETRetriever"),
    }
    target = lazy.get(name)
    if target is None:
        raise AttributeError(f"module 'graphknows.retrieval' has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


__all__ = [
    "RetrievalContext",
    "build_retriever",
]
