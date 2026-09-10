"""T070/FR-032/FR-033: a query term that names an entity, not a concept.

Mocked client, no ArcadeDB: what is pinned is that an identifier-shaped term
survives tokenising whole, is matched against ``ENTITY.name_norm`` through the
same normalisation ingest writes, and pulls back the facts that entity is the
subject of.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from graphknows.retrieval.retriever import SymbolRecall
from graphknows.storage.arcadedb._base import ArcadeStoreBase

_CONCEPTS = [{"uri": "demo:Person", "label": "Person"}]


def _row(fact_id: str, name_norm: str) -> dict[str, Any]:
    """One activation row for a fact the named entity is the subject of."""
    return {
        "id": fact_id,
        "subject": "e1",
        "predicate": "demo:imports",
        "object": "e2",
        "polarity": "asserted",
        "modality": "",
        "confidence": 1.0,
        "valid_from": "",
        "valid_to": "",
        "text": "graphknows/memory.py imports the retriever.",
        "byte_start": 0,
        "byte_end": 43,
        "source_uri": "file:///notes.md",
        "name_norm": name_norm,
    }


def _store(named: list[dict[str, Any]]) -> ArcadeStoreBase:
    """A store answering the concept lookup and the name_norm activation."""

    async def answer(_db: str, cypher: str, params: dict | None = None) -> list[dict]:
        if "name_norm" in cypher:
            return named
        if "EVOKES" in cypher or "INSTANCE_OF" in cypher or "SAME_AS_CANDIDATE" in cypher:
            return []
        return _CONCEPTS

    client = MagicMock()
    client.query = AsyncMock(side_effect=answer)
    return ArcadeStoreBase(client, "db")


def _queries(store: ArcadeStoreBase) -> str:
    return "\n".join(call.args[1] for call in store.client.query.await_args_list)


async def test_an_identifier_term_activates_the_facts_its_entity_is_subject_of() -> None:
    store = _store([_row("f1", "graphknows/memory.py")])

    result = await SymbolRecall(store).recall("what changed in graphknows/memory.py?")

    text = _queries(store)
    assert "e.name_norm IN ['graphknows/memory.py']" in text
    assert "(e)-[:SUBJECT_OF]->(f:FACT)-[:ASSERTED_IN]->(g:SEGMENT)" in text
    assert [item.fact.id for item in result.facts] == ["f1"]
    assert result.no_evidence is False
    assert result.counters.symbols_resolved == 1


async def test_identifier_shapes_are_kept_whole_and_normalised() -> None:
    store = _store([])

    await SymbolRecall(store).recall("Does name_norm in GraphKnows/Memory.py call Store.query?")

    assert "['graphknows/memory.py','name_norm','store.query']" in _queries(store)


async def test_a_plain_word_query_asks_no_entity_question() -> None:
    store = _store([])

    result = await SymbolRecall(store).recall("what did the person build")

    assert "name_norm" not in _queries(store)
    assert result.counters.symbols_resolved == 1


async def test_an_unmatched_identifier_counts_as_an_unresolved_symbol() -> None:
    result = await SymbolRecall(_store([])).recall("graphknows/nowhere.py")

    assert result.no_evidence is True
    assert result.counters.symbols_resolved == 0
    assert result.counters.symbols_unresolved >= 1


async def test_a_tombstoned_entity_is_excluded_in_sql() -> None:
    store = _store([])

    await SymbolRecall(store).recall("name_norm")

    assert "e.state <> 'forgotten'" in _queries(store)
