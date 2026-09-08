"""Lazy settings + the process runtime accessor for MCP tool modules.

One MCP server process serves exactly one namespace, bound at startup from
``settings.namespace`` (``GRAPHKNOWS_NAMESPACE``). Serving several tenants means
running one process per tenant: no caller can name a namespace, so no caller can
reach another tenant's database.
"""

from __future__ import annotations

import asyncio
from typing import Any

from graphknows.settings import get_settings

_runtime: Any | None = None
_lock = asyncio.Lock()

__all__ = ["close", "get_or_init_retriever", "get_runtime", "get_settings"]


async def get_runtime() -> Any:
    """Return the process ``Memory``, creating and caching it lazily.

    Concurrent first-use callers build it exactly once (guarded).
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _lock:
        if _runtime is None:
            from graphknows.memory import Memory

            _runtime = Memory(get_settings(), namespace=get_settings().namespace)
        return _runtime


async def get_or_init_retriever() -> Any:
    """Return the retriever via the process runtime."""
    return await (await get_runtime())._get_retriever()


async def close() -> None:
    """Close the cached runtime (e.g. in tests / shutdown)."""
    import contextlib

    global _runtime
    async with _lock:
        if _runtime is not None:
            with contextlib.suppress(Exception):
                await _runtime.close()
        _runtime = None
        get_settings.cache_clear()
