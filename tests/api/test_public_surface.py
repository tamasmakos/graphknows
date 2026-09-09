"""Snapshot of the public API surface.

The documented public surface of ``graphknows`` is exactly ``graphknows.__all__``
plus one subpackage per integration under ``graphknows.integrations.*``, the two
CLI entry points, and the MCP tool contract. This test pins the top-level export
set so an accidental addition or removal is a deliberate, reviewed change (bump
the snapshot in the same commit).
"""

from __future__ import annotations

import graphknows

EXPECTED_PUBLIC_API = {
    "MAX_CONTEXT_CHARS",
    "ConfigurationError",
    "GraphKnowsError",
    "GraphKnowsSettings",
    # The recall result type and its renderer. Public because every caller of
    # Memory.recall_memory needs them: the hits it returns ARE Hit objects, and
    # rendering them (dating, attributing, deduping, budgeting) is the library's
    # job — four adapters re-derived it from an untyped dict before, and
    # disagreed.
    "Hit",
    "IngestResult",
    "Memory",
    "MemoryMode",
    "MemoryScope",
    "Message",
    "MissingExtraError",
    "RenderedMemories",
    "StoreError",
    "TopicMode",
    "__version__",
    "date_in",
    "normalize_messages",
    "render_memories",
}


def test_public_all_matches_snapshot() -> None:
    assert set(graphknows.__all__) == EXPECTED_PUBLIC_API


def test_every_exported_name_is_importable() -> None:
    for name in graphknows.__all__:
        assert hasattr(graphknows, name), f"{name} is in __all__ but not importable"


def test_memory_is_the_facade() -> None:
    from graphknows.memory import Memory as RuntimeMemory

    assert graphknows.Memory is RuntimeMemory
