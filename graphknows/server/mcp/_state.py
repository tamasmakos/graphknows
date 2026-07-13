"""Lazy settings + per-namespace runtime accessors for MCP tool modules.

The MCP server is one process that can serve many tenants concurrently. Each
*namespace* (see :mod:`graphknows.storage.namespace`) is a hard isolation boundary
backed by its own physical ArcadeDB database pair, so we keep one ``Memory`` per
namespace and route each tool call to it. Namespaces resolve as:
explicit tool argument → ``GRAPHKNOWS_NAMESPACE`` (settings default) → ``""``.

The cache is bounded: past ``_MAX_OPEN_NAMESPACES`` live runtimes the
least-recently-used one is closed, so unbounded tenant cardinality cannot leak
httpx connection pools.
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from typing import Any

from graphknows.settings import GraphKnowsSettings

_settings: GraphKnowsSettings | None = None
# LRU cache: namespace → Memory. Ordered so the front is least-recently-used.
_runtimes: OrderedDict[str, Any] = OrderedDict()
_lock = asyncio.Lock()

# Cap on concurrently-open namespace runtimes. Configurable so a deployment
# serving many tenants can raise it to match its ArcadeDB server sizing.
_MAX_OPEN_NAMESPACES = int(os.environ.get("GRAPHKNOWS_MAX_OPEN_NAMESPACES", "64"))


def get_settings() -> GraphKnowsSettings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def _resolve(namespace: str) -> str:
    """Effective namespace: explicit arg wins, else the settings default."""
    return namespace or get_settings().namespace


async def get_runtime(namespace: str = "") -> Any:
    """Return the ``Memory`` for *namespace*, creating and caching it lazily.

    Concurrent first-use callers for the same namespace build it exactly once
    (guarded). Evicts (closes) the least-recently-used runtime past the cap.
    """
    ns = _resolve(namespace)
    runtime = _runtimes.get(ns)
    if runtime is not None:
        _runtimes.move_to_end(ns)  # mark most-recently-used
        return runtime
    async with _lock:
        runtime = _runtimes.get(ns)
        if runtime is not None:
            _runtimes.move_to_end(ns)
            return runtime
        from graphknows.memory import Memory

        runtime = Memory(get_settings(), namespace=ns)
        _runtimes[ns] = runtime
        _runtimes.move_to_end(ns)
        await _evict_if_needed()
        return runtime


async def _evict_if_needed() -> None:
    """Close least-recently-used runtimes until under the open cap. Caller holds _lock."""
    import contextlib

    while len(_runtimes) > _MAX_OPEN_NAMESPACES:
        _evicted_ns, evicted = _runtimes.popitem(last=False)  # least-recently-used
        with contextlib.suppress(Exception):
            await evicted.close()


async def get_or_init_retriever(namespace: str = "") -> Any:
    """Return the retriever for *namespace* via its shared runtime."""
    return await (await get_runtime(namespace))._get_retriever()


async def close() -> None:
    """Close all cached namespace runtimes (e.g. in tests / shutdown)."""
    import contextlib

    global _settings
    async with _lock:
        for runtime in _runtimes.values():
            with contextlib.suppress(Exception):
                await runtime.close()
        _runtimes.clear()
        _settings = None
