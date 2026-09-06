"""Concurrent ``add()`` calls on one session must not collide on ``TURN.seq``.

The buffer numbers its turns by reading ``max(t.seq)`` back out of the store and
adding one. That read-then-write is not atomic on its own: two coroutines that
interleave both see the same maximum, both write ``<session>:<n>``, and the
second ``MERGE`` overwrites the first — a turn is silently lost, and the class
docstring's "safe for concurrent asyncio use" is false. This pins the
per-session lock in ``Memory.add`` against a live ArcadeDB.

Nothing is stubbed: the conversation path writes plain TURN vertices, so no
extraction or embedding runs here anyway. ``embed_dimensions`` is set only to
keep schema creation from probing the embedding model.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings, MemoryMode

_CALLS = 8


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_add_allocates_distinct_turn_seqs(arcadedb_required: None) -> None:
    """Eight simultaneous single-message adds leave eight turns on seq 1..8."""
    settings = GraphKnowsSettings().model_copy(update={"embed_dimensions": 8})
    memory = Memory(
        settings=settings,
        namespace=f"test_{uuid.uuid4().hex[:8]}",
        mode=MemoryMode.llm_free,
    )
    session_id = "sess-concurrent-add"
    try:
        await memory._get_retriever()  # connect once, so the race is on seq alone
        await asyncio.gather(
            *(
                memory.add([{"role": "user", "content": f"message {i}"}], run_id=session_id)
                for i in range(_CALLS)
            )
        )

        rows = await memory._store.query(
            "MATCH (t:TURN {session_id: $sid}) RETURN t.seq AS seq, t.text AS text",
            sid=session_id,
        )
        assert sorted(int(r["seq"]) for r in rows) == list(range(1, _CALLS + 1))
        assert len({str(r["text"]) for r in rows}) == _CALLS, "a turn was overwritten"
    finally:
        try:
            await memory._store.drop_database()
        finally:
            await memory.close()
