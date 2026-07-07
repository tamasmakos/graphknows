"""Snapshot of the public API surface.

The documented public surface of ``graphknows`` is exactly ``graphknows.__all__``
plus ``graphknows.ports``, ``graphknows.client``, ``graphknows.adapters.*``, the
two CLI entry points, and the MCP tool contract. This test pins the top-level
export set so an accidental addition or removal is a deliberate, reviewed change
(bump the snapshot in the same commit).
"""

from __future__ import annotations

import importlib
import warnings

import graphknows

EXPECTED_PUBLIC_API = {
    "ConfigurationError",
    "ExtractionError",
    "GraphKnowsError",
    "GraphKnowsSettings",
    "IngestError",
    "IngestResult",
    "Memory",
    "MemoryMode",
    "MissingExtraError",
    "RetrievalError",
    "StoreError",
    "__version__",
}


def test_public_all_matches_snapshot() -> None:
    assert set(graphknows.__all__) == EXPECTED_PUBLIC_API


def test_every_exported_name_is_importable() -> None:
    for name in graphknows.__all__:
        assert hasattr(graphknows, name), f"{name} is in __all__ but not importable"


def test_memory_is_the_facade() -> None:
    from graphknows.runtime import Memory as RuntimeMemory

    assert graphknows.Memory is RuntimeMemory


def test_graphknowsruntime_alias_is_deprecated() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = graphknows.GraphKnowsRuntime
    assert alias is graphknows.Memory
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_ports_are_a_public_subpackage() -> None:
    ports = importlib.import_module("graphknows.ports")
    for name in ("IEmbedder", "IRelationExtractor", "IVectorStore", "ISTMStore", "ILTMStore"):
        assert hasattr(ports, name)
