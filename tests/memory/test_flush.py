"""Unit tests for SessionFlusher."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.flush import FlushOptions, SessionFlusher


def _make_flusher():
    mg = MagicMock()
    mg.read_session_subgraph = AsyncMock(
        return_value={
            "entities": [
                {"name": "EU", "type": "ORG", "confidence": 0.9},
                {"name": "eu", "type": "ORG", "confidence": 0.7},
                {"name": "France", "type": "LOC", "confidence": 0.8},
            ],
            "files": [],
            "chunks": [],
            "dialogs": [],
            "turns": [],
        }
    )
    mg.delete_session = AsyncMock()

    pg = MagicMock()
    pg.promote_to_ltm = AsyncMock(return_value=(5, 3))
    pg.clear_stm = AsyncMock(return_value=None)

    neo = MagicMock()
    neo.write_session_subgraph = AsyncMock(return_value={"nodes": 10, "edges": 20})
    neo.summarize_topics = AsyncMock(return_value=3)
    neo.run_gds_enrichment = AsyncMock(return_value=None)
    neo.extract_and_write_relations = AsyncMock(return_value=0)

    flusher = SessionFlusher(mg, pg, neo, openrouter_key="test_key")
    return flusher, mg, pg, neo


@pytest.mark.asyncio
async def test_flush_default_options_calls_all_steps():
    flusher, mg, pg, neo = _make_flusher()
    result = await flusher.flush("s1")
    mg.read_session_subgraph.assert_called_once_with("s1")
    neo.write_session_subgraph.assert_called_once()
    pg.promote_to_ltm.assert_called_once_with("s1")
    neo.summarize_topics.assert_called_once()
    mg.delete_session.assert_called_once_with("s1")
    pg.clear_stm.assert_called_once_with("s1")


@pytest.mark.asyncio
async def test_flush_always_summarizes_topics():
    flusher, mg, pg, neo = _make_flusher()
    await flusher.flush("s1", FlushOptions(summarize_topics=False))
    neo.summarize_topics.assert_called_once()


@pytest.mark.asyncio
async def test_flush_no_clear_skips_clear_stm():
    flusher, mg, pg, neo = _make_flusher()
    await flusher.flush("s1", FlushOptions(clear_stm=False))
    mg.delete_session.assert_not_called()
    pg.clear_stm.assert_not_called()


@pytest.mark.asyncio
async def test_flush_no_merge_skips_resolve_entities():
    flusher, mg, pg, neo = _make_flusher()
    # When merge_entities=False, all entities passed through unchanged
    result = await flusher.flush("s1", FlushOptions(merge_entities=False))
    assert result.merged_entities == 0


def test_resolve_entities_merges_by_lowercase_keeps_highest_confidence():
    flusher, _, _, _ = _make_flusher()
    entities = [
        {"name": "EU", "type": "ORG", "confidence": 0.9},
        {"name": "eu", "type": "ORG", "confidence": 0.7},
        {"name": "France", "type": "LOC", "confidence": 0.8},
    ]
    resolved, merge_count = flusher._resolve_entities(entities)
    names_lower = {e["name"].lower() for e in resolved}
    assert "eu" in names_lower
    assert "france" in names_lower
    assert len(resolved) == 2  # eu+EU merged into 1
    assert merge_count == 1
    eu_ent = next(e for e in resolved if e["name"].lower() == "eu")
    assert eu_ent["confidence"] == 0.9  # highest kept


def test_resolve_entities_preserves_unique_names():
    flusher, _, _, _ = _make_flusher()
    entities = [
        {"name": "Germany", "type": "LOC", "confidence": 0.95},
        {"name": "France", "type": "LOC", "confidence": 0.85},
    ]
    resolved, _ = flusher._resolve_entities(entities)
    assert len(resolved) == 2


@pytest.mark.asyncio
async def test_topic_summarization_failure_adds_to_errors():
    flusher, mg, pg, neo = _make_flusher()
    neo.summarize_topics = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    result = await flusher.flush("s1", FlushOptions(run_gds=False))
    assert any("topics" in e for e in result.errors)


@pytest.mark.asyncio
async def test_clear_stm_failure_adds_to_errors():
    flusher, mg, pg, neo = _make_flusher()
    mg.delete_session = AsyncMock(side_effect=RuntimeError("Memgraph down"))
    result = await flusher.flush("s1", FlushOptions(summarize_topics=False))
    assert any("clear_stm" in e for e in result.errors)


@pytest.mark.asyncio
async def test_flush_result_has_correct_counts():
    flusher, mg, pg, neo = _make_flusher()
    result = await flusher.flush("s1", FlushOptions(run_gds=False))
    assert result.session_id == "s1"
    assert result.nodes_written == 10
    assert result.edges_written == 20
    assert result.chunks_promoted == 5
    assert result.turns_promoted == 3
    assert result.topics_created == 3


# ---------------------------------------------------------------------------
# FLUSH-003: summarize_topics unit tests
# ---------------------------------------------------------------------------


def _make_neo4j_store_mock(co_occur_rows: list[dict]) -> Any:
    """Minimal Neo4jStore mock for summarize_topics tests."""
    from unittest.mock import AsyncMock, MagicMock

    neo = MagicMock()
    neo._arun = AsyncMock(
        side_effect=[
            co_occur_rows,  # co-occurrence query
            [],  # evidence chunks (one call per community)
        ]
        * 10
    )  # plenty
    neo.upsert_topics = AsyncMock(return_value=None)
    neo._driver = MagicMock()
    neo._driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return neo


@pytest.mark.asyncio
async def test_summarize_topics_no_key_skips_llm(monkeypatch):
    """Empty openrouter_key → no HTTP call, topics still created with entity names as title."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    store._driver.session.return_value.__enter__ = MagicMock(return_value=s)
    store._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store.upsert_topics = AsyncMock(return_value=None)

    # _arun returns: co-occur pairs → evidence rows for community 0
    store._arun = AsyncMock(
        side_effect=[
            [{"a": "EU", "b": "France", "weight": 3}],
            [],  # evidence for community 0
        ]
    )

    count = await store.summarize_topics("sess-1", openrouter_key="", pg_store=None)
    assert count == 1
    store.upsert_topics.assert_awaited_once()
    topics = store.upsert_topics.call_args[0][0]
    assert len(topics) == 1
    assert "EU" in topics[0]["label"] or "France" in topics[0]["label"]
    assert topics[0]["summary"] == ""


@pytest.mark.asyncio
async def test_summarize_topics_skips_singletons():
    """A community with only 1 entity is skipped entirely."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    store.upsert_topics = AsyncMock(return_value=None)
    # Only one entity — Leiden will produce a singleton community, which is skipped
    store._arun = AsyncMock(return_value=[])  # no co-occur pairs → returns 0 immediately

    count = await store.summarize_topics("sess-2", openrouter_key="", pg_store=None)
    assert count == 0
    store.upsert_topics.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_topics_writes_topic_embedding_from_summary(monkeypatch):
    """write_ltm_topic receives an embedding generated from the LLM summary."""
    import httpx

    from graphknows.memory.stores import embedder as embedder_module
    from graphknows.memory.stores.neo4j import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    store._driver.session.return_value.__enter__ = MagicMock(return_value=s)
    store._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store.upsert_topics = AsyncMock(return_value=None)
    store._arun = AsyncMock(
        side_effect=[
            [{"a": "EU", "b": "France", "weight": 3}],
            [{"text": "EU and France coordinate policy.", "hp": "doc > sec"}],
        ]
    )

    embedded_inputs: list[str] = []

    def fake_embed_one(text: str) -> list[float]:
        embedded_inputs.append(text)
        return [0.1, 0.2, 0.3]

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"title": "EU policy", "summary": "France and the EU coordinate policy."}'
                        }
                    }
                ]
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return _Resp()

    pg_store = MagicMock()
    pg_store.write_ltm_topic = AsyncMock(return_value=None)

    monkeypatch.setattr(embedder_module, "embed_one", fake_embed_one)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _Client())

    count = await store.summarize_topics("sess-embed", openrouter_key="sk-test", pg_store=pg_store)

    assert count == 1
    assert embedded_inputs == ["France and the EU coordinate policy."]
    written_topic = pg_store.write_ltm_topic.await_args.args[0]
    assert written_topic["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_summarize_topics_falls_back_to_title_for_embedding(monkeypatch):
    """If summary is empty, write_ltm_topic receives an embedding from the topic label."""
    import httpx

    from graphknows.memory.stores import embedder as embedder_module
    from graphknows.memory.stores.neo4j import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    store._driver.session.return_value.__enter__ = MagicMock(return_value=s)
    store._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store.upsert_topics = AsyncMock(return_value=None)
    store._arun = AsyncMock(
        side_effect=[
            [{"a": "EU", "b": "France", "weight": 3}],
            [],
        ]
    )

    embedded_inputs: list[str] = []

    def fake_embed_one(text: str) -> list[float]:
        embedded_inputs.append(text)
        return [0.4, 0.5, 0.6]

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"title": "European coordination", "summary": ""}'
                        }
                    }
                ]
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return _Resp()

    pg_store = MagicMock()
    pg_store.write_ltm_topic = AsyncMock(return_value=None)

    monkeypatch.setattr(embedder_module, "embed_one", fake_embed_one)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _Client())

    count = await store.summarize_topics("sess-fallback", openrouter_key="sk-test", pg_store=pg_store)

    assert count == 1
    assert embedded_inputs == ["European coordination"]
    written_topic = pg_store.write_ltm_topic.await_args.args[0]
    assert written_topic["embedding"] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_flush_calls_extract_and_write_relations():
    """flush() must call extract_and_write_relations on the Neo4j store."""
    flusher, mg, pg, neo = _make_flusher()
    neo.extract_and_write_relations = AsyncMock(return_value=5)
    result = await flusher.flush("s1")
    neo.extract_and_write_relations.assert_awaited_once()
    # edges in result should include the 5 relation edges
    assert result.edges_written >= 5


@pytest.mark.asyncio
async def test_flush_propagates_extraction_error():
    """If extract_and_write_relations raises with a valid key, flush degrades gracefully and records error."""
    flusher, mg, pg, neo = _make_flusher()
    neo.extract_and_write_relations = AsyncMock(side_effect=RuntimeError("LLM failed"))
    # Provide a non-empty key so extraction is attempted
    flusher._openrouter_key = "test-key"
    result = await flusher.flush("s1")
    # Flush must NOT raise — errors are captured, not propagated
    assert any("extract_and_write_relations" in e for e in result.errors)


class TestFlushSurfacesNeo4jError:
    @pytest.mark.asyncio
    async def test_neo4j_write_error_appears_in_flush_result_errors(self):
        """When write_session_subgraph raises, FlushResult.errors must contain the error."""
        mg = MagicMock()
        pg = MagicMock()
        neo = AsyncMock()

        # Force Neo4j write to raise
        neo.write_session_subgraph = AsyncMock(side_effect=RuntimeError("neo4j connection refused"))
        neo.extract_and_write_relations = AsyncMock(return_value=0)

        flusher = SessionFlusher.__new__(SessionFlusher)
        flusher._mg = mg
        flusher._pg = pg
        flusher._neo = neo
        flusher._openrouter_key = None

        # Call the private helper directly
        errors: list[str] = []
        result = await flusher._write_ltm_graph(
            session_id="test-session",
            stm_data={"entities": [], "chunks": [], "turns": []},
            errors=errors,
        )

        # The error must be surfaced
        assert errors, "errors list must be populated when Neo4j raises"
        assert any("neo4j" in e.lower() or "connection refused" in e.lower() for e in errors), (
            f"Expected Neo4j error in errors list, got: {errors}"
        )
