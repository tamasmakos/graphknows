"""Unit tests for ADAPT-001: AgentMemoryAdapter protocol + GraphknowsMemoryAdapter."""

from __future__ import annotations

from typing import Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.adapter import (
    AgentMemoryAdapter,
    FlushResult,
    GraphknowsMemoryAdapter,
    IngestResult,
    MemoryStats,
    RecallHit,
)

# ---------------------------------------------------------------------------
# DTO tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ingest_result_dataclass():
    r = IngestResult(file_id="f1", chunks=3, entities=5, elapsed_s=1.2)
    assert r.file_id == "f1"
    assert r.chunks == 3
    assert r.entities == 5
    assert r.elapsed_s == 1.2
    assert r.errors == []


@pytest.mark.unit
def test_recall_hit_dataclass():
    h = RecallHit(
        score=0.95,
        text="some text",
        session_id="s1",
        entities=["e1", "e2"],
        source_type="stm_chunk",
    )
    assert h.score == 0.95
    assert h.text == "some text"
    assert h.session_id == "s1"
    assert h.entities == ["e1", "e2"]
    assert h.source_type == "stm_chunk"


@pytest.mark.unit
def test_flush_result_dataclass():
    r = FlushResult(nodes=10, edges=5, chunks=3, turns=2, elapsed_s=2.5)
    assert r.nodes == 10
    assert r.edges == 5
    assert r.chunks == 3
    assert r.turns == 2
    assert r.elapsed_s == 2.5
    assert r.errors == []


@pytest.mark.unit
def test_memory_stats_dataclass():
    s = MemoryStats(stm={"chunks": 10}, ltm={"entities": 5})
    assert s.stm == {"chunks": 10}
    assert s.ltm == {"entities": 5}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_graphknows_memory_adapter_satisfies_protocol():
    adapter = GraphknowsMemoryAdapter()
    assert isinstance(adapter, AgentMemoryAdapter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recall_result(score, text, source_type="chunk", entities=None):
    from graphknows.retrieval.recall_stm import RecallResult

    return RecallResult(
        text=text,
        source_type=source_type,
        scope="short_term",
        score=score,
        entity_names=entities or [],
    )


async def _sync_to_thread(fn, *args, **kwargs):
    """Replacement for ``asyncio.to_thread`` that runs synchronously."""
    return fn(*args, **kwargs)


@pytest.fixture
def mock_container():
    c = MagicMock()
    c.memgraph = MagicMock()
    c.pg = MagicMock()
    c.neo = MagicMock()
    return c


# ---------------------------------------------------------------------------
# GraphknowsMemoryAdapter — ingest
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_delegates_to_memory_ingestor(mock_container):
    internal_result = MagicMock()
    internal_result.file_id = "f1"
    internal_result.chunk_count = 3
    internal_result.entity_count = 5
    internal_result.elapsed_s = 1.2
    internal_result.errors = []

    mock_ingestor = MagicMock()
    mock_ingestor.ingest_text = MagicMock(return_value=internal_result)

    with (
        patch(
            "graphknows.memory.extractor.GLiNER2EntityExtractor",
            return_value=MagicMock(),
        ),
        patch(
            "graphknows.memory.ingestor.MemoryIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "graphknows.memory.adapter.asyncio.to_thread",
            new=_sync_to_thread,
        ),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.ingest(
            text="hello world",
            session_id="s1",
            title="doc1",
        )

    assert isinstance(result, IngestResult)
    assert result.file_id == "f1"
    assert result.chunks == 3
    assert result.entities == 5


# ---------------------------------------------------------------------------
# GraphknowsMemoryAdapter — recall
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_stm(mock_container):
    with (
        patch(
            "graphknows.retrieval.recall_stm.recall_short_term",
            return_value=[
                _make_recall_result(0.9, "stm hit", entities=["e1"]),
            ],
        ),
        patch(
            "graphknows.memory.adapter.asyncio.to_thread",
            new=_sync_to_thread,
        ),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(
            query="test", session_id="s1", top_k=5, scope="stm"
        )

    assert len(result) == 1
    assert isinstance(result[0], RecallHit)
    assert result[0].score == 0.9
    assert result[0].text == "stm hit"
    assert result[0].entities == ["e1"]
    assert result[0].session_id == "s1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_ltm(mock_container):
    with (
        patch(
            "graphknows.retrieval.recall_stm.recall_short_term",
            return_value=[],
        ),
        patch(
            "graphknows.retrieval.recall_ltm.recall_long_term",
            return_value=[
                _make_recall_result(0.8, "ltm hit", source_type="turn"),
            ],
        ),
        patch(
            "graphknows.memory.adapter.asyncio.to_thread",
            new=_sync_to_thread,
        ),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(
            query="test", session_id="s1", top_k=5, scope="ltm"
        )

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].text == "ltm hit"
    assert result[0].source_type == "turn"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_both(mock_container):
    with (
        patch(
            "graphknows.retrieval.recall_stm.recall_short_term",
            return_value=[
                _make_recall_result(0.9, "stm hit", entities=["e1"]),
            ],
        ),
        patch(
            "graphknows.retrieval.recall_ltm.recall_long_term",
            return_value=[
                _make_recall_result(0.7, "ltm hit", source_type="turn"),
            ],
        ),
        patch(
            "graphknows.memory.adapter.asyncio.to_thread",
            new=_sync_to_thread,
        ),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(
            query="test", session_id="s1", top_k=5, scope="both"
        )

    assert len(result) == 2
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# GraphknowsMemoryAdapter — flush
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_flush_calls_session_flusher(mock_container):
    internal = MagicMock()
    internal.nodes_written = 10
    internal.edges_written = 5
    internal.chunks_promoted = 3
    internal.turns_promoted = 2
    internal.elapsed_s = 1.5
    internal.errors = []

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=internal)

    with patch(
        "graphknows.memory.flush.SessionFlusher",
        return_value=mock_flusher,
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.flush(session_id="s1")

    assert isinstance(result, FlushResult)
    assert result.nodes == 10
    assert result.edges == 5
    assert result.chunks == 3
    assert result.turns == 2


# ---------------------------------------------------------------------------
# GraphknowsMemoryAdapter — stats via container
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_stats_queries_pg_and_neo4j(mock_container):
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(side_effect=[10, 5, 20, 15, 3])
    mock_conn.close = AsyncMock()

    mock_neo_session = MagicMock()
    mock_neo_session.run.side_effect = [
        MagicMock(single=lambda: {"c": 42}),
        MagicMock(single=lambda: {"c": 7}),
    ]
    mock_neo_session.__enter__ = MagicMock(return_value=mock_neo_session)
    mock_neo_session.__exit__ = MagicMock(return_value=None)
    mock_neo_driver = MagicMock()
    mock_neo_driver.session = MagicMock(return_value=mock_neo_session)

    with (
        patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)),
        patch(
            "neo4j.GraphDatabase.driver",
            return_value=mock_neo_driver,
        ),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.stats()

    assert isinstance(result, MemoryStats)
    assert result.stm["chunks"] == 10
    assert result.stm["turns"] == 5
    assert result.ltm["chunks"] == 20
    assert result.ltm["entities"] == 42
    assert result.ltm["sessions"] == 7


# ---------------------------------------------------------------------------
# Protocol type-check
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_agent_memory_adapter_is_protocol():
    assert hasattr(AgentMemoryAdapter, "__protocol__") or issubclass(
        AgentMemoryAdapter, Protocol
    )
