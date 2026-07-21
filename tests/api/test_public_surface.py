"""Snapshot of the public API surface.

The documented public surface of ``graphknows`` is exactly ``graphknows.__all__``
plus ``graphknows.ports``, ``graphknows.integrations.client``, ``graphknows.integrations.*``, the
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
    "MemoryScope",
    "Message",
    "MissingExtraError",
    "RetrievalError",
    "StoreError",
    "__version__",
    "normalize_messages",
}


def test_public_all_matches_snapshot() -> None:
    assert set(graphknows.__all__) == EXPECTED_PUBLIC_API


def test_every_exported_name_is_importable() -> None:
    for name in graphknows.__all__:
        assert hasattr(graphknows, name), f"{name} is in __all__ but not importable"


def test_memory_is_the_facade() -> None:
    from graphknows.memory import Memory as RuntimeMemory

    assert graphknows.Memory is RuntimeMemory


def test_store_protocols_are_public() -> None:
    base = importlib.import_module("graphknows.storage.base")
    for name in ("IEmbedder", "IGraphStore", "ISTMStore", "ILTMStore"):
        assert hasattr(base, name)
    relations = importlib.import_module("graphknows.ingestion.extraction.relations.base")
    assert hasattr(relations, "IRelationExtractor")
