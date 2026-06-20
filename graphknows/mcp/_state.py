"""Lazy settings + runtime accessors for MCP tool modules."""

from __future__ import annotations

from typing import Any

from graphknows.config import GraphKnowsSettings

_settings: GraphKnowsSettings | None = None
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


async def get_or_init_retriever() -> Any:
    """Return the singleton retriever via the shared runtime."""
    return await get_runtime()._get_retriever()


async def close() -> None:
    """Close all store connections held by the singleton runtime."""
    global _runtime, _settings
    if _runtime is not None:
        await _runtime.close()
    _runtime = None
    _settings = None
