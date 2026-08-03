"""Integration tests for the TURN short-term-memory buffer.

Conversation turns land as cheap ``TURN`` vertices at ``add()`` time (no
extraction, no embedding, no CHUNK nodes). ``flush()`` runs the buffer through
the real ingestion pipeline and flips it raw -> consolidated. A plain-string
``add()`` is unchanged: it still goes through the eager document path.

These tests exercise only the ``Memory`` public seam (``add``/``search``/
``flush``/``stats``) against a live ArcadeDB, each in its own throwaway
namespace so runs don't interfere and cleanup is a single ``drop_namespace``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/mem",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def mem() -> AsyncIterator[object]:
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.memory import Memory
    from graphknows.settings import GraphKnowsSettings

    namespace = f"test-turnbuf-{uuid.uuid4().hex[:8]}"
    m = Memory(settings=GraphKnowsSettings(), namespace=namespace)
    try:
        yield m
    finally:
        await m.drop_namespace()
        await m.close()


async def test_conversation_add_writes_turns_not_chunks(mem: object) -> None:
    res = await mem.add([{"role": "user", "content": "hello there"}], run_id="s1")

    assert res["turns"] == 1
    assert res["chunks"] == 0
    assert res["entities"] == 0

    stats = await mem.stats()
    assert stats["turns"]["raw"] == 1
    assert stats["chunks"]["raw"] == 0


async def test_document_add_still_produces_chunks_immediately(mem: object) -> None:
    res = await mem.add(
        "Photosynthesis converts sunlight into chemical energy in plants.", run_id="s2"
    )

    assert res["chunks"] > 0

    stats = await mem.stats()
    assert stats["chunks"]["raw"] == res["chunks"]


async def test_search_stm_finds_buffered_turn_before_flush(mem: object) -> None:
    await mem.add(
        [{"role": "user", "content": "the quokka festival begins tomorrow at noon"}],
        run_id="s3",
    )

    out = await mem.search("quokka festival", run_id="s3", scope="stm")

    assert len(out["hits"]) == 1
    assert out["hits"][0].sources == "stm"
    assert "quokka" in out["hits"][0].text.lower()


async def test_search_ltm_is_empty_before_flush(mem: object) -> None:
    await mem.add(
        [{"role": "user", "content": "the quokka festival begins tomorrow at noon"}],
        run_id="s4",
    )

    out = await mem.search("quokka festival", run_id="s4", scope="ltm")

    assert out["hits"] == []


async def test_flush_drains_buffer_into_the_graph(mem: object) -> None:
    await mem.add(
        [{"role": "user", "content": "the quokka festival begins tomorrow at noon"}],
        run_id="s5",
    )

    flushed = await mem.flush(run_id="s5")
    assert flushed["turns"] == 1
    assert flushed["errors"] == []

    stm_out = await mem.search("quokka festival", run_id="s5", scope="stm")
    assert stm_out["hits"] == []

    ltm_out = await mem.search("quokka festival", run_id="s5", scope="ltm")
    assert len(ltm_out["hits"]) >= 1
    assert any("quokka" in h.text.lower() for h in ltm_out["hits"])


async def test_stats_raw_turn_count_grows_per_add(mem: object) -> None:
    # stats() queries the namespace database directly (no ensure_schema), so it
    # cannot run before the first add() lazily creates it.
    await mem.add([{"role": "user", "content": "first turn"}], run_id="s6")
    stats1 = await mem.stats()
    assert stats1["turns"]["raw"] == 1

    await mem.add([{"role": "user", "content": "second turn"}], run_id="s6")
    stats2 = await mem.stats()
    assert stats2["turns"]["raw"] == 2


async def test_per_turn_infer_and_speaker_survive_the_flush(mem) -> None:
    """The buffer must not flatten the caller's per-turn decisions.

    Extraction policy belongs to the caller, not the library: it says which
    turns are worth extracting from (infer) and who said them (Message.name).
    The buffer defers ingestion to flush, so both have to be carried across
    that boundary — otherwise flush re-ingests every turn under one shared flag
    and the caller's per-turn choice is silently lost.
    """
    # A sentence spaCy's NER actually fires on: extraction short-circuits to an
    # empty result whenever the chunk mines no entity label at all.
    await mem.add(
        [
            {
                "role": "user",
                "name": "Gina",
                "content": "Mochi's vet is Dr. Nagy, her clinic is in Budapest.",
            }
        ],
        run_id="s1",
        infer=True,
    )
    await mem.add(
        [{"role": "user", "name": "Jon", "content": "Hey Gina, good to see you!"}],
        run_id="s1",
        infer=False,
    )

    await mem.flush(run_id="s1")

    hits = await mem.search("Mochi vet", run_id="s1", top_k=10, scope="ltm")
    assert hits["hits"], "extracted turn must still be retrievable"

    # Both turns are stored and retrievable; only the infer=True one has entities.
    entities = await mem.ltm_entities(limit=200)
    names = {str(e.get("name", "")).casefold() for e in entities.get("entities", [])}
    assert any("mochi" in n for n in names), f"expected Mochi among {sorted(names)[:20]}"
    assert not any("hey gina" in n for n in names), (
        f"infer=False turn must mint no entities, got {sorted(names)[:20]}"
    )
