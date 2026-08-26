"""Writing a turn through GraphKnowsMemory, without reaching past the seam.

A caller with a structured turn — a speaker, a timestamp, and its own decision
about whether the turn is worth extracting from — had no method to call.
``remember`` reads a framework's message state and cannot carry ``infer``, so
the only route was ``gm.memory.add(run_id=..., infer=...)``: through the escape
hatch, and having learned how a run_id becomes a session id on the way.
"""

from __future__ import annotations

import pytest

from graphknows.integrations.langgraph import GraphKnowsMemory
from tests.fixtures.memory import FakeMemory as _FakeMemory


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_buffers_a_structured_turn_into_this_session(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("Mochi's vet is Dr. Nagy", speaker="Gina", timestamp="20 January, 2023")

        assert mem.ingested == [
            (
                "s1",
                [
                    {
                        "role": "user",
                        "content": "Mochi's vet is Dr. Nagy",
                        "name": "Gina",
                        "timestamp": "20 January, 2023",
                    }
                ],
            )
        ]

    @pytest.mark.asyncio
    async def test_add_carries_the_callers_extraction_decision(self) -> None:
        """``infer`` is the caller's call — the library has no opinion on it."""
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("Hey Jon!", infer=False)

        assert mem.ingest_kwargs[0]["infer"] is False

    @pytest.mark.asyncio
    async def test_add_defaults_to_extracting(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("Mochi is a ragdoll")

        assert mem.ingest_kwargs[0]["infer"] is True

    @pytest.mark.asyncio
    async def test_add_omits_absent_metadata_rather_than_sending_blanks(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("hello")

        _, messages = mem.ingested[0]
        assert messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_add_takes_a_role_for_the_agents_own_turns(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("Got it, a ragdoll.", role="assistant")

        _, messages = mem.ingested[0]
        assert messages[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_add_ignores_an_empty_utterance(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem)

        await gm.add("   ")

        assert mem.ingested == []


class TestInferPolicy:
    """``infer`` may be a predicate asked per turn, set once for the session.

    Without it, a caller wanting "extract from assertions, not greetings" had to
    abandon ``remember``, drive ``ingest_memory`` by hand, and re-derive the
    decision in its own graph node — which is exactly what the LoCoMo runner did.
    """

    @pytest.mark.asyncio
    async def test_a_predicate_is_asked_per_turn(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem, infer=lambda text: "vet" in text)

        await gm.add("Hey Jon!")
        await gm.add("Mochi's vet is Dr. Nagy")

        assert [kw["infer"] for kw in mem.ingest_kwargs] == [False, True]

    @pytest.mark.asyncio
    async def test_the_predicate_also_governs_the_remember_node(self) -> None:
        """Both write paths, one policy — or the graph node silently ignores it."""
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem, infer=lambda text: "vet" in text)

        await gm.remember({"messages": [{"role": "user", "content": "Hey Jon!"}]})

        assert mem.ingest_kwargs[0]["infer"] is False

    @pytest.mark.asyncio
    async def test_a_per_turn_argument_overrides_the_session_policy(self) -> None:
        mem = _FakeMemory()
        gm = GraphKnowsMemory("s1", memory=mem, infer=lambda text: False)

        await gm.add("Mochi is a ragdoll", infer=True)

        assert mem.ingest_kwargs[0]["infer"] is True


class TestUnflushedWarning:
    """Forgetting ``flush`` yields an empty graph and no error to notice.

    A buffered turn mints nothing until flush drains it, so a caller that exits
    without flushing gets a memory that answers nothing — indistinguishable
    from a broken install unless the exit says so.
    """

    @pytest.mark.asyncio
    async def test_exiting_with_buffered_turns_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mem = _FakeMemory()
        with caplog.at_level("WARNING", logger="graphknows.integrations.langgraph._session"):
            async with GraphKnowsMemory("s1", memory=mem) as gm:
                await gm.add("Mochi is a ragdoll")

        assert "s1" in caplog.text
        assert "flush" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_exiting_after_a_flush_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        mem = _FakeMemory()
        with caplog.at_level("WARNING", logger="graphknows.integrations.langgraph._session"):
            async with GraphKnowsMemory("s1", memory=mem) as gm:
                await gm.add("Mochi is a ragdoll")
                await gm.flush()

        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_a_turn_that_buffered_nothing_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``remember`` is a no-op on a turn with no text — so it must not count.

        Counting node calls rather than stored turns warns "1 unflushed turn"
        over an empty buffer, which is a false alarm on the one signal that
        exists to catch a genuinely unflushed session.
        """
        mem = _FakeMemory()
        with caplog.at_level("WARNING", logger="graphknows.integrations.langgraph._session"):
            async with GraphKnowsMemory("s1", memory=mem) as gm:
                await gm.remember({"messages": [{"role": "user", "content": ""}]})
                await gm.add("   ")

        assert mem.ingested == []
        assert caplog.text == ""
