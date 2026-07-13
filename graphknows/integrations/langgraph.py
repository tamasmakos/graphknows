"""LangGraph integration — framework-agnostic node hooks + a langgraph store.

``recall``/``remember`` are duck-typed node helpers (they take a ``Memory`` and
touch nothing from langgraph), so importing them is free. ``GraphKnowsStore``
subclasses ``langgraph.store.base.BaseStore`` and is resolved lazily so importing
this module does not require langgraph to be installed.
"""

from __future__ import annotations

from typing import Any

from graphknows.integrations._langgraph_hooks import recall, remember

__all__ = ["GraphKnowsStore", "recall", "remember"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``GraphKnowsStore`` (imports langgraph on first access)."""
    if name == "GraphKnowsStore":
        from graphknows.integrations._langgraph_store import GraphKnowsStore

        return GraphKnowsStore
    raise AttributeError(f"module 'graphknows.integrations.langgraph' has no attribute {name!r}")
