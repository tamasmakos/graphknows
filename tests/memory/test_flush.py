"""Unit tests for LTMPromotionPipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.flush import LTMPromotionPipeline
from graphknows.memory.models.promotion import PromotionRecord, PromotionRequest


def _make_pipeline(**kwargs):
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
    neo.extract_and_write_relations = AsyncMock(return_value=0)
    neo._driver = MagicMock()

    enrichment = MagicMock()
    from graphknows.memory.models.gds import GDSResult
    enrichment.run_gds_async = AsyncMock(return_value=GDSResult())
    enrichment.run_topic_summarization = AsyncMock(return_value=3)

    pipeline = LTMPromotionPipeline(
        mg, pg, neo, openrouter_key="test_key", enrichment_service=enrichment, **kwargs
    )
    return pipeline, mg, pg, neo, enrichment


@pytest.mark.asyncio
async def test_promote_default_request_calls_all_steps():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    result = await pipeline.promote(PromotionRequest(session_id="s1"))
    mg.read_session_subgraph.assert_called_once_with("s1")
    neo.write_session_subgraph.assert_called_once()
    pg.promote_to_ltm.assert_called_once_with("s1")
    enrichment.run_topic_summarization.assert_called_once()
    mg.delete_session.assert_called_once_with("s1")
    pg.clear_stm.assert_called_once_with("s1")


@pytest.mark.asyncio
async def test_promote_no_clear_skips_clear_stm():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    await pipeline.promote(PromotionRequest(session_id="s1", clear_stm=False))
    mg.delete_session.assert_not_called()
    pg.clear_stm.assert_not_called()


@pytest.mark.asyncio
async def test_promote_no_merge_skips_resolve_entities():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    result = await pipeline.promote(PromotionRequest(session_id="s1", merge_entities=False))
    assert result.merged_entities == 0


def test_resolve_entities_merges_by_lowercase_keeps_highest_confidence():
    pipeline, _, _, _, _ = _make_pipeline()
    entities = [
        {"name": "EU", "type": "ORG", "confidence": 0.9},
        {"name": "eu", "type": "ORG", "confidence": 0.7},
        {"name": "France", "type": "LOC", "confidence": 0.8},
    ]
    resolved, merge_count = pipeline._resolve_entities(entities)
    names_lower = {e["name"].lower() for e in resolved}
    assert "eu" in names_lower
    assert "france" in names_lower
    assert len(resolved) == 2
    assert merge_count == 1
    eu_ent = next(e for e in resolved if e["name"].lower() == "eu")
    assert eu_ent["confidence"] == 0.9


def test_resolve_entities_preserves_unique_names():
    pipeline, _, _, _, _ = _make_pipeline()
    entities = [
        {"name": "Germany", "type": "LOC", "confidence": 0.95},
        {"name": "France", "type": "LOC", "confidence": 0.85},
    ]
    resolved, _ = pipeline._resolve_entities(entities)
    assert len(resolved) == 2


@pytest.mark.asyncio
async def test_topic_summarization_failure_adds_to_errors():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    enrichment.run_topic_summarization = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    result = await pipeline.promote(PromotionRequest(session_id="s1", run_gds=False))
    assert any("topics" in e for e in result.errors)


@pytest.mark.asyncio
async def test_clear_stm_failure_adds_to_errors():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    mg.delete_session = AsyncMock(side_effect=RuntimeError("Memgraph down"))
    result = await pipeline.promote(PromotionRequest(session_id="s1"))
    assert any("clear_stm" in e for e in result.errors)


@pytest.mark.asyncio
async def test_promote_result_has_correct_counts():
    pipeline, mg, pg, neo, enrichment = _make_pipeline()
    result = await pipeline.promote(PromotionRequest(session_id="s1", run_gds=False))
    assert result.session_id == "s1"
    assert result.nodes_written == 10
    assert result.edges_written == 20
    assert result.chunks_promoted == 5
    assert result.turns_promoted == 3
    assert result.topics_created == 3


# ---------------------------------------------------------------------------
# FLUSH-003: summarize_topics unit tests (unchanged — test Neo4jStore directly)
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

    store._arun = AsyncMock(
        side_effect=[
            [],
            [{"a": "EU", "b": "France", "weight": 3}],
            [],
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
    store._arun = AsyncMock(return_value=[])

    count = await store.summarize_topics("sess-2", openrouter_key="", pg_store=None)
    assert count == 0
    store.upsert_topics.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_topics_writes_topic_embedding_from_summary(monkeypatch):
    """write_ltm_topic receives an embedding generated from the LLM summary."""
    from graphknows.memory.stores import embedder as embedder_module
    from graphknows.memory.stores.neo4j import Neo4jStore
    from graphknows.memory.ltm.topic_summarizer import TopicSummarizer

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    store._driver.session.return_value.__enter__ = MagicMock(return_value=s)
    store._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store.upsert_topics = AsyncMock(return_value=None)
    store._arun = AsyncMock(
        side_effect=[
            [],
            [{"a": "EU", "b": "France", "weight": 3}],
            [],
            [{"text": "EU and France coordinate policy.", "hp": "doc > sec"}],
        ]
    )

    embedded_inputs: list[str] = []

    def fake_embed_one(text: str) -> list[float]:
        embedded_inputs.append(text)
        return [0.1, 0.2, 0.3]

    pg_store = MagicMock()
    pg_store.write_ltm_topic = AsyncMock(return_value=None)

    async def fake_summarize(self, *, members, passages, fallback_title):
        return "EU policy", "France and the EU coordinate policy."

    monkeypatch.setattr(embedder_module, "embed_one", fake_embed_one)
    monkeypatch.setattr(TopicSummarizer, "summarize", fake_summarize)

    count = await store.summarize_topics("sess-embed", openrouter_key="sk-test", pg_store=pg_store)

    assert count == 1
    assert embedded_inputs == ["France and the EU coordinate policy."]
    written_topic = pg_store.write_ltm_topic.await_args.args[0]
    assert written_topic["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_summarize_topics_falls_back_to_title_for_embedding(monkeypatch):
    """If summary is empty, write_ltm_topic receives an embedding from the topic label."""
    from graphknows.memory.stores import embedder as embedder_module
    from graphknows.memory.stores.neo4j import Neo4jStore
    from graphknows.memory.ltm.topic_summarizer import TopicSummarizer

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = MagicMock()
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    store._driver.session.return_value.__enter__ = MagicMock(return_value=s)
    store._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store.upsert_topics = AsyncMock(return_value=None)
    store._arun = AsyncMock(
        side_effect=[
            [],
            [{"a": "EU", "b": "France", "weight": 3}],
            [],
            [],
        ]
    )

    embedded_inputs: list[str] = []

    def fake_embed_one(text: str) -> list[float]:
        embedded_inputs.append(text)
        return [0.4, 0.5, 0.6]

    pg_store = MagicMock()
    pg_store.write_ltm_topic = AsyncMock(return_value=None)

    async def fake_summarize(self, *, members, passages, fallback_title):
        return "European coordination", ""

    monkeypatch.setattr(embedder_module, "embed_one", fake_embed_one)
    monkeypatch.setattr(TopicSummarizer, "summarize", fake_summarize)

    count = await store.summarize_topics("sess-fallback", openrouter_key="sk-test", pg_store=pg_store)

    assert count == 1
    assert embedded_inputs == ["European coordination"]
    written_topic = pg_store.write_ltm_topic.await_args.args[0]
    assert written_topic["embedding"] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_promote_calls_extract_and_write_relations():
    """promote() must call extract_and_write_relations when mode is llm_assisted."""
    pipeline, mg, pg, neo, enrichment = _make_pipeline(relation_extraction_mode="llm_assisted")
    neo.extract_and_write_relations = AsyncMock(return_value=5)
    result = await pipeline.promote(PromotionRequest(session_id="s1"))
    neo.extract_and_write_relations.assert_awaited_once()
    assert result.edges_written >= 5


@pytest.mark.asyncio
async def test_promote_propagates_extraction_error():
    """If extract_and_write_relations raises, promote degrades gracefully and records error."""
    pipeline, mg, pg, neo, enrichment = _make_pipeline(relation_extraction_mode="llm_assisted")
    neo.extract_and_write_relations = AsyncMock(side_effect=RuntimeError("LLM failed"))
    result = await pipeline.promote(PromotionRequest(session_id="s1"))
    assert any("extract_and_write_relations" in e for e in result.errors)


class TestPromoteSurfacesNeo4jError:
    @pytest.mark.asyncio
    async def test_neo4j_write_error_appears_in_record_errors(self):
        """When write_session_subgraph raises, PromotionRecord.errors must contain the error."""
        mg = MagicMock()
        pg = MagicMock()
        neo = AsyncMock()

        neo.write_session_subgraph = AsyncMock(side_effect=RuntimeError("neo4j connection refused"))
        neo.extract_and_write_relations = AsyncMock(return_value=0)

        pipeline = LTMPromotionPipeline.__new__(LTMPromotionPipeline)
        pipeline._mg = mg
        pipeline._pg = pg
        pipeline._neo = neo
        pipeline._openrouter_key = None
        pipeline._relation_extraction_mode = "gliner2"

        errors: list[str] = []
        result = await pipeline._write_ltm_graph(
            session_id="test-session",
            stm_data={"entities": [], "chunks": [], "turns": []},
            errors=errors,
        )

        assert errors, "errors list must be populated when Neo4j raises"
        assert any("neo4j" in e.lower() or "connection refused" in e.lower() for e in errors), (
            f"Expected Neo4j error in errors list, got: {errors}"
        )
