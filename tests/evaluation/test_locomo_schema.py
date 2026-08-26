"""The LoCoMo-specific schema layer, built on the library's channel seam.

Everything asserted here is a property of the DATASET, not of conversation
memory: one date per session, a fixed speaker set, dates that only the session
knows. That is the dividing line the seam exists to hold — the library learns
none of it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.scripts.locomo_langgraph import (
    LoCoMoSessionSchema,
    session_dates,
)

_SESSIONS = [
    {
        "session_id": "conv-30-s1",
        "messages": [
            {"role": "Gina", "content": "Hey Jon!", "timestamp": "4:04 pm on 20 January, 2023"},
            {"role": "Jon", "content": "Lost my job.", "timestamp": "4:04 pm on 20 January, 2023"},
        ],
    },
    {
        "session_id": "conv-30-s2",
        "messages": [
            {"role": "Gina", "content": "Hi again", "timestamp": "2:32 pm on 29 January, 2023"}
        ],
    },
]


def _store() -> MagicMock:
    store = MagicMock()
    store.command = AsyncMock(return_value=[])
    store.query = AsyncMock(return_value=[])
    store.client = MagicMock()
    store.client.command = AsyncMock(return_value=[])
    store.database = "mem_conv_30"
    return store


def _cyphers(store: MagicMock) -> list[str]:
    return [c.args[0] for c in store.command.await_args_list]


class TestSessionDates:
    def test_one_date_per_session_at_day_granularity(self) -> None:
        """A LoCoMo session is a conversation-DAY: every turn shares its stamp."""
        assert session_dates(_SESSIONS) == {
            "conv-30-s1": "2023-01-20",
            "conv-30-s2": "2023-01-29",
        }

    def test_a_session_with_no_parseable_stamp_is_absent(self) -> None:
        """Absent, not defaulted: a wrong date is worse than a missing one."""
        assert session_dates([{"session_id": "s9", "messages": [{"content": "hi"}]}]) == {}


class TestPopulate:
    @pytest.mark.asyncio
    async def test_declares_its_own_edge_type(self) -> None:
        """The layer owns its DDL; ArcadeDB needs the type before the edge."""
        store = _store()

        await LoCoMoSessionSchema({"s1": "2023-01-20"}).populate(store, "s1")

        ddl = [c.args[1] for c in store.client.command.await_args_list]
        assert "CREATE EDGE TYPE ON_DATE IF NOT EXISTS" in ddl

    @pytest.mark.asyncio
    async def test_links_the_session_to_its_date_node(self) -> None:
        store = _store()

        await LoCoMoSessionSchema({"s1": "2023-01-20"}).populate(store, "s1")

        on_date = [c for c in store.command.await_args_list if "ON_DATE" in c.args[0]]
        assert len(on_date) == 1
        assert on_date[0].kwargs["iso"] == "2023-01-20"
        assert "MERGE (d:TEMPORAL {date: $iso})" in on_date[0].args[0]

    @pytest.mark.asyncio
    async def test_backfills_only_chunks_that_have_no_date(self) -> None:
        """The turn's own stamp is more precise; overwriting flattens a session.

        This is the measurable half: 26 of conv-30's 81 questions want a date as
        the ANSWER, and a chunk that renders undated cannot supply one.
        """
        store = _store()

        await LoCoMoSessionSchema({"s1": "2023-01-20"}).populate(store, "s1")

        backfill = [c for c in store.command.await_args_list if "SET c.ts" in c.args[0]]
        assert len(backfill) == 1
        assert "c.ts IS NULL OR c.ts = ''" in backfill[0].args[0]

    @pytest.mark.asyncio
    async def test_writes_no_attribution_edge(self) -> None:
        """The layer writes only what it reads back.

        It used to also write CHUNK-[:SAID_BY]->ENTITY on every flush, which
        nothing ever traversed — attribution already reaches the answerer as
        ``CHUNK.speaker`` via ``Hit.speaker``. Write-only schema is the failure
        this seam exists to make visible, so the edge is gone rather than
        wired up.
        """
        store = _store()

        await LoCoMoSessionSchema({"s1": "2023-01-20"}).populate(store, "s1")

        assert not any("SAID_BY" in c for c in _cyphers(store))

    @pytest.mark.asyncio
    async def test_a_session_with_no_known_date_writes_nothing(self) -> None:
        store = _store()

        await LoCoMoSessionSchema({}).populate(store, "unknown-session")

        assert not any("ON_DATE" in c for c in _cyphers(store))
        assert not any("SET c.ts" in c for c in _cyphers(store))


class TestCollect:
    @staticmethod
    def _ctx(query: str) -> Any:
        ctx = MagicMock()
        ctx.query = query
        return ctx

    @pytest.mark.asyncio
    async def test_reaches_chunks_whose_session_knows_the_date(self) -> None:
        """The complement of the built-in temporal channel.

        That one walks CHUNK-[:MENTIONS_DATE]->TEMPORAL, so it finds only chunks
        that SAY the date. The turn answering "what did Gina find on 1 February,
        2023?" never writes the date down — only its session knows it.
        """
        rt = MagicMock()
        rt.store = MagicMock()
        rt.store.query = AsyncMock(return_value=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])

        out = await LoCoMoSessionSchema({}).collect(
            self._ctx("What did Gina find on 1 February, 2023?"), rt, 10
        )

        assert list(out) == ["c1", "c2"]
        assert out["c1"][0] > out["c2"][0], "must be ranked, not flat"
        assert out["c1"][1]["sources"] == {"locomo_session"}
        cypher = rt.store.query.await_args.args[0]
        assert "[:ON_DATE]" in cypher and "[:IN_SESSION]" in cypher

    @pytest.mark.asyncio
    async def test_a_query_naming_no_date_collects_nothing(self) -> None:
        """No date, no work: 79 of conv-30's 81 questions take this path."""
        rt = MagicMock()
        rt.store = MagicMock()
        rt.store.query = AsyncMock(return_value=[])

        out = await LoCoMoSessionSchema({}).collect(self._ctx("Why did Jon quit?"), rt, 10)

        assert out == {}
        rt.store.query.assert_not_awaited()
