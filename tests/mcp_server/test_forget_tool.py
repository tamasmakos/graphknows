"""T031: ``memory_forget`` over the facade, and the segment cascade behind it.

Forget is a tombstone (FR-037): nothing is deleted, so a merge-log replay still
reads the record while every recall path filters it out. Forgetting a segment
cascades to the facts it was the only evidence for (edge case 8), and the tool
hands the caller the counters — ``facts_excluded_tombstoned`` — rather than a
bare "ok".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from graphknows.memory import Memory
from graphknows.server.mcp.tools.ltm import memory_forget


@dataclass
class _FakeStore:
    """Records every write; answers the cascade with *cascaded* facts."""

    cascaded: int = 0
    commands: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def command(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.commands.append((cypher, params))
        return [{"facts": self.cascaded}] if "ASSERTED_IN" in cypher else []


def _connected(store: _FakeStore) -> Memory:
    memory = Memory(namespace="t031-forget")
    memory._retriever = object()
    memory._store = store
    return memory


@pytest.fixture
def store() -> Any:
    fake = _FakeStore()

    async def _get_runtime() -> Memory:
        return _connected(fake)

    with patch("graphknows.server.mcp.tools.ltm.get_runtime", _get_runtime):
        yield fake


@pytest.mark.asyncio
async def test_forget_tombstones_the_record_without_deleting_it(store: _FakeStore) -> None:
    await memory_forget("f1")

    tombstoned = [c for c, p in store.commands if p.get("state") == "forgotten"]
    assert len(tombstoned) == 4  # FACT, ENTITY, SEGMENT, and the cascade
    assert all("DELETE" not in cypher for cypher, _ in store.commands)


@pytest.mark.asyncio
async def test_forgetting_a_segment_cascades_to_the_facts_it_evidenced(
    store: _FakeStore,
) -> None:
    """Only facts left without live evidence go; the filter is the shared fragment."""
    store.cascaded = 2

    out = await memory_forget("seg1")

    cascade = next(c for c, _ in store.commands if "ASSERTED_IN" in c)
    assert "state <> 'forgotten'" in cascade
    assert out["facts_excluded_tombstoned"] == 2


@pytest.mark.asyncio
async def test_a_fact_with_other_evidence_is_not_counted(store: _FakeStore) -> None:
    out = await memory_forget("seg1")

    assert out["facts_excluded_tombstoned"] == 0


@pytest.mark.asyncio
async def test_the_tool_is_registered_and_takes_no_namespace() -> None:
    """Per-fact forget is a user operation, and no tool names a namespace."""
    from graphknows.server.mcp import tools
    from graphknows.server.mcp._app import app

    tool = next(t for t in await app.list_tools() if t.name == "memory_forget")
    assert "memory_forget" in tools.__all__
    assert set(tool.inputSchema["properties"]) == {"record_id"}
