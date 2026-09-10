"""T071: ``memory_recall`` must be an adapter over the facade, not a second retriever.

The tool hands back a ``RecallResult`` as JSON — facts with the evidence that
asserts them (FR-009, SC-002), and an explicit ``no_evidence`` where nothing is
known rather than an empty list presented as success (FR-015).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from graphknows.models.fact import Fact
from graphknows.models.report import (
    Counters,
    Evidence,
    FactWithEvidence,
    RecallBudget,
    RecallResult,
)
from graphknows.server.mcp.tools.recall import memory_recall

SLICE = RecallResult(
    facts=[
        FactWithEvidence(
            fact=Fact(id="f1", subject="e1", predicate="p1", object="e2"),
            evidence=[
                Evidence(
                    source_uri="file:///repo/a.py", byte_range=(0, 12), text="Acme is Platinum"
                )
            ],
        )
    ],
    truncated_by="confidence",
    counters=Counters(facts_returned=1),
)


@dataclass
class _FakeRuntime:
    """Records what the facade was asked for; answers with *result*."""

    result: RecallResult = SLICE
    calls: list[tuple[str, RecallBudget | None]] = field(default_factory=list)

    async def recall(self, query: str, budget: RecallBudget | None = None) -> RecallResult:
        self.calls.append((query, budget))
        return self.result


@pytest.fixture
def runtime() -> Any:
    fake = _FakeRuntime()

    async def _get_runtime() -> _FakeRuntime:
        return fake

    with patch("graphknows.server.mcp.tools.recall.get_runtime", _get_runtime):
        yield fake


@pytest.mark.asyncio
async def test_facts_come_back_with_their_evidence(runtime: _FakeRuntime) -> None:
    out = await memory_recall("what about Acme?")

    assert runtime.calls == [("what about Acme?", None)]
    (fact,) = out["facts"]
    assert fact["fact"]["id"] == "f1"
    assert fact["evidence"] == [
        {"source_uri": "file:///repo/a.py", "byte_range": [0, 12], "text": "Acme is Platinum"}
    ]
    assert out["no_evidence"] is False
    assert out["truncated_by"] == "confidence"
    assert out["budget"] == {"max_facts": 20, "max_evidence_per_fact": 1}
    assert out["counters"]["facts_returned"] == 1


@pytest.mark.asyncio
async def test_the_budget_reaches_the_facade(runtime: _FakeRuntime) -> None:
    """An explicit bound is the caller's (FR-010), not the tool's to reinterpret."""
    await memory_recall("Acme", budget={"max_facts": 3, "max_evidence_per_fact": 2})

    assert runtime.calls == [("Acme", RecallBudget(max_facts=3, max_evidence_per_fact=2))]


@pytest.mark.asyncio
async def test_nothing_known_is_stated_not_implied(runtime: _FakeRuntime) -> None:
    runtime.result = RecallResult(no_evidence=True, counters=Counters(symbols_unresolved=1))

    out = await memory_recall("nothing here resolves")

    assert out["facts"] == []
    assert out["no_evidence"] is True


@pytest.mark.asyncio
async def test_the_tool_is_registered_and_takes_no_namespace() -> None:
    from graphknows.server.mcp import tools
    from graphknows.server.mcp._app import app

    tool = next(t for t in await app.list_tools() if t.name == "memory_recall")
    assert "memory_recall" in tools.__all__
    assert set(tool.inputSchema["properties"]) == {"query", "budget"}
