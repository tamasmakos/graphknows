"""T027/FR-002: top-down activation, and the budget it answers under.

Mocked client, no ArcadeDB: what is pinned is that recall reads back the two
edges bottom-up labelling wrote, that every returned fact carries its evidence,
and that an empty or cut slice says so instead of returning a bare list.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.models.report import RecallBudget
from graphknows.retrieval.retriever import SymbolRecall
from graphknows.storage.arcadedb._base import ArcadeStoreBase

_CONCEPTS = [{"uri": "demo:Person", "label": "Person"}]


def _row(fact_id: str, confidence: float = 1.0, **over: Any) -> dict[str, Any]:
    """One activation row: an assertion and the segment it was read from."""
    return {
        "id": fact_id,
        "subject": "e1",
        "predicate": "demo:built",
        "object": "e2",
        "polarity": "asserted",
        "modality": "",
        "confidence": confidence,
        "valid_from": "",
        "valid_to": "",
        "text": "Ada designed the Engine.",
        "byte_start": 0,
        "byte_end": 24,
        "source_uri": "file:///notes.md",
        **over,
    }


def _store(evoked: list[dict], instantiated: list[dict] | None = None) -> ArcadeStoreBase:
    """A store answering the concept lookup and the two activation queries."""

    async def answer(_db: str, cypher: str, params: dict | None = None) -> list[dict]:
        if "EVOKES" in cypher:
            return evoked
        if "INSTANCE_OF" in cypher:
            return instantiated or []
        return _CONCEPTS

    client = MagicMock()
    client.query = AsyncMock(side_effect=answer)
    return ArcadeStoreBase(client, "db")


def _queries(store: ArcadeStoreBase) -> str:
    return "\n".join(call.args[1] for call in store.client.query.await_args_list)


async def test_activation_reads_evokes_and_instance_of() -> None:
    store = _store([_row("f1")], [_row("f2")])

    result = await SymbolRecall(store).recall("what did the person build")

    text = _queries(store)
    assert "(g:SEGMENT)-[:EVOKES]->(c:CONCEPT)" in text
    assert "(e:ENTITY)-[:INSTANCE_OF]->(c:CONCEPT)" in text
    assert "'demo:Person'" in text
    assert {item.fact.id for item in result.facts} == {"f1", "f2"}
    assert result.counters.symbols_resolved == 1
    assert result.counters.facts_returned == 2


async def test_every_returned_fact_carries_its_evidence() -> None:
    result = await SymbolRecall(_store([_row("f1")])).recall("person")

    [item] = result.facts
    [evidence] = item.evidence
    assert (evidence.source_uri, evidence.byte_range) == ("file:///notes.md", (0, 24))
    assert evidence.text == "Ada designed the Engine."
    assert item.fact.predicate == "demo:built"
    assert result.no_evidence is False


async def test_a_tombstoned_fact_is_excluded_in_sql() -> None:
    store = _store([])

    await SymbolRecall(store).recall("person")

    assert _queries(store).count("f.state <> 'forgotten'") == 2


async def test_an_unresolved_query_says_nothing_is_known() -> None:
    store = _store([_row("f1")])

    result = await SymbolRecall(store).recall("submarine")

    assert result.no_evidence is True and result.facts == []
    assert (result.counters.symbols_resolved, result.counters.symbols_unresolved) == (0, 1)
    assert "EVOKES" not in _queries(store)


async def test_activated_symbols_reaching_nothing_say_nothing_is_known() -> None:
    result = await SymbolRecall(_store([])).recall("person")

    assert result.no_evidence is True and result.counters.symbols_resolved == 1


async def test_the_budget_cuts_by_a_stated_ordering() -> None:
    rows = [_row("f1", 0.4), _row("f2", 0.9), _row("f3", 0.6)]

    result = await SymbolRecall(_store(rows)).recall("person", RecallBudget(max_facts=2))

    assert [item.fact.id for item in result.facts] == ["f2", "f3"]
    assert result.truncated_by == "fact confidence"
    assert result.counters.facts_truncated_by_budget == 1
    assert result.budget.max_facts == 2


async def test_an_uncut_slice_states_no_truncation() -> None:
    result = await SymbolRecall(_store([_row("f1")])).recall("person")

    assert result.truncated_by is None and result.counters.facts_truncated_by_budget == 0


async def test_evidence_per_fact_is_bounded_and_deduplicated() -> None:
    same_segment = _row("f1")
    other_segment = _row("f1", text="Later, she wrote the notes.", byte_start=25, byte_end=52)
    store = _store([same_segment, other_segment], [same_segment])

    result = await SymbolRecall(store).recall("person", RecallBudget(max_evidence_per_fact=2))

    [item] = result.facts
    assert [evidence.byte_range for evidence in item.evidence] == [(0, 24), (25, 52)]


@pytest.mark.parametrize("budget", [RecallBudget(max_facts=1), RecallBudget()])
async def test_recall_never_returns_a_fact_without_evidence(budget: RecallBudget) -> None:
    result = await SymbolRecall(_store([_row("f1")])).recall("person", budget)

    assert all(item.evidence for item in result.facts)
