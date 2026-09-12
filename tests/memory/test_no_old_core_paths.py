"""After cutover the facade ingests through the pipeline only (FR-045, FR-046)."""

from __future__ import annotations

import inspect

import graphknows.memory as memory

_OLD_PATHS = (
    "write_turn",
    "list_turns",
    "next_turn_seq",
    "mark_turns_consolidated",
    "compute_and_persist_pagerank",
    "compute_graph_embeddings",
    "compute_communities",
    "consolidate_session",
    "consolidate_relations",
)


def test_facade_has_no_old_core_paths() -> None:
    source = inspect.getsource(memory)
    leaked = [name for name in _OLD_PATHS if name in source]
    assert not leaked, f"facade still routes through the old core: {leaked}"
