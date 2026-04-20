"""
Plugin system for GraphKnows graph post-processing.

Plugins are auto-discovered: any GraphPlugin subclass in the
services/graphgen/src/kg/plugins/ package is registered automatically.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type["GraphPlugin"]] = {}


class GraphPlugin(ABC):
    """Base class for all graph post-processing plugins."""

    #: Override in subclasses — used as the registry key.
    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            _REGISTRY[cls.name] = cls
            logger.debug("Registered graph plugin: %s", cls.name)

    @abstractmethod
    async def run(self, graph: nx.DiGraph, **kwargs: Any) -> nx.DiGraph:
        """Apply plugin transformations to *graph* and return the modified graph."""


def get_plugin(name: str) -> GraphPlugin:
    """Instantiate a registered plugin by name."""
    if name not in _REGISTRY:
        raise KeyError(f"No graph plugin registered as '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def all_plugins() -> list[GraphPlugin]:
    """Return instances of all registered plugins."""
    return [cls() for cls in _REGISTRY.values()]


def _autodiscover() -> None:
    """Import every module in the plugins package so subclasses register themselves."""
    import kg.plugins as pkg  # noqa: PLC0415

    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        if module_name.startswith("_"):
            continue
        full_name = f"kg.plugins.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load plugin module %s: %s", full_name, exc)


_autodiscover()
