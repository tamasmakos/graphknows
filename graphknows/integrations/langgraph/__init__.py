"""LangGraph integration — framework-agnostic node hooks + a langgraph store.

``recall``/``remember`` are duck-typed node helpers (they take a ``Memory`` and
touch nothing from langgraph), so importing them is free. ``GraphKnowsMemory``
is a session-bound convenience wrapper over the same hooks — also free to
import, it pulls no langgraph either. ``GraphKnowsStore`` subclasses
``langgraph.store.base.BaseStore`` and is resolved lazily so importing this
module does not require langgraph to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.integrations.langgraph._hooks import recall, remember
from graphknows.integrations.langgraph._session import GraphKnowsMemory

if TYPE_CHECKING:
    # Declares the name for ruff/mypy without importing langgraph at runtime —
    # `__getattr__` below is what actually resolves it. Without this the entry
    # in `__all__` is an undefined export (F822) to every static analyser.
    from graphknows.integrations.langgraph._store import GraphKnowsStore

__all__ = ["GraphKnowsMemory", "GraphKnowsStore", "recall", "remember"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``GraphKnowsStore`` (imports langgraph on first access)."""
    if name == "GraphKnowsStore":
        from graphknows.integrations.langgraph._store import GraphKnowsStore

        return GraphKnowsStore
    raise AttributeError(f"module 'graphknows.integrations.langgraph' has no attribute {name!r}")
