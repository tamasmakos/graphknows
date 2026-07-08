"""LangGraph reference adapter for graphknows memory.

- :func:`recall` / :func:`remember` — framework-agnostic node helpers (no
  langgraph dependency; usable from any ``state -> partial_state`` graph).
- :class:`GraphKnowsStore` — a langgraph ``BaseStore`` backed by graphknows.
  Requires the ``langgraph`` extra; imported lazily so the helpers above work
  without it.
"""

from __future__ import annotations

from typing import Any

from graphknows.adapters.langgraph.hooks import recall, remember

# GraphKnowsStore is intentionally NOT in __all__: it requires the ``langgraph``
# extra, so keeping it out of __all__ lets ``from ... import *`` pull the
# pure-Python helpers without triggering the langgraph import. Explicit
# ``from graphknows.adapters.langgraph import GraphKnowsStore`` still resolves it
# (and raises MissingExtraError with an install hint when langgraph is absent).
__all__ = ["recall", "remember"]


def __getattr__(name: str) -> Any:
    if name == "GraphKnowsStore":
        from graphknows.adapters.langgraph.store import GraphKnowsStore

        return GraphKnowsStore
    raise AttributeError(f"module 'graphknows.adapters.langgraph' has no attribute {name!r}")
