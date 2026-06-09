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
    from graphknows.memory.models.recall import RecallResult

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
            "graphknows.memory.stm.recall.recall_short_term",
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
        result = await adapter.recall(query="test", session_id="s1", top_k=5, scope="stm")

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
            "graphknows.memory.stm.recall.recall_short_term",
            return_value=[],
        ),
        patch(
            "graphknows.memory.query.recall_ltm.recall_long_term",
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
        result = await adapter.recall(query="test", session_id="s1", top_k=5, scope="ltm")

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].text == "ltm hit"
    assert result[0].source_type == "turn"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_both(mock_container):
    with (
        patch(
            "graphknows.memory.stm.recall.recall_short_term",
            return_value=[
                _make_recall_result(0.9, "stm hit", entities=["e1"]),
            ],
        ),
        patch(
            "graphknows.memory.query.recall_ltm.recall_long_term",
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
        result = await adapter.recall(query="test", session_id="s1", top_k=5, scope="both")

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

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=internal)

    with patch(
        "graphknows.memory.flush.LTMPromotionPipeline",
        return_value=mock_pipeline,
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

    mock_container.neo._arun = AsyncMock(
        side_effect=[[{"c": 42}], [{"c": 7}]]
    )

    with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
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
    assert hasattr(AgentMemoryAdapter, "__protocol__") or issubclass(AgentMemoryAdapter, Protocol)


# ===========================================================================
# GraphknowsAdapter full lifecycle
# ===========================================================================


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.adapter import GraphknowsAdapter
from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    MemoryClass,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)
from graphknows.memory.models.recall import RecallResult


def _make_adapter():
    mg = MagicMock()
    pg = MagicMock()
    neo = MagicMock()
    extractor = MagicMock()
    embed_fn = MagicMock(return_value=[0.1] * 10)
    return GraphknowsAdapter(mg, pg, neo, extractor, embed_fn)


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
def test_open_session_sets_active_session_id(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    adapter.open_session("s1")
    assert adapter._active_session_id == "s1"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
def test_ingest_artifact_delegates_to_ingestor(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    artifact = ArtifactRecord(
        artifact_id="a1",
        path="/tmp/test.py",
        content="print('hello')",
        mime_type="text/x-python",
        memory_class=MemoryClass.raw_evidence,
    )
    adapter.ingest_artifact(artifact, "s1")

    mock_ingestor.return_value.ingest_text.assert_called_once()
    args, kwargs = mock_ingestor.return_value.ingest_text.call_args
    assert args[0] == "print('hello')"
    assert kwargs.get("session_id") == "s1"
    assert kwargs.get("title") == "/tmp/test.py"
    assert kwargs.get("source_hint") == "text/x-python"
    metadata = kwargs.get("metadata", {})
    assert metadata.get("memory_class") == "raw_evidence"
    assert metadata.get("artifact_id") == "a1"


@patch("graphknows.memory.stm.recall.recall_short_term")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
def test_recall_returns_hits_as_dicts(
    mock_dialog_writer,
    mock_ingestor,
    mock_recall,
):
    adapter = _make_adapter()
    mock_recall.return_value = [
        RecallResult(
            text="hello",
            source_type="chunk",
            scope="short_term",
            score=0.9,
        )
    ]
    request = RecallRequest(query="test", session_id="s1", top_k=5)
    result = adapter.recall(request)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "hello"
    assert result[0]["source_type"] == "chunk"
    assert result[0]["scope"] == "short_term"


@patch("graphknows.memory.stm.recall.recall_short_term")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
def test_recall_filters_raw_evidence_when_include_raw_false(
    mock_dialog_writer,
    mock_ingestor,
    mock_recall,
):
    adapter = _make_adapter()
    mock_recall.return_value = [
        RecallResult(
            text="raw",
            source_type="chunk",
            scope="short_term",
            score=0.9,
            provenance={"memory_class": "raw_evidence"},
        ),
        RecallResult(
            text="summary",
            source_type="chunk",
            scope="short_term",
            score=0.8,
            provenance={"memory_class": "summary"},
        ),
    ]
    request = RecallRequest(
        query="test",
        session_id="s1",
        top_k=5,
        include_raw=False,
    )
    result = adapter.recall(request)

    assert len(result) == 1
    assert result[0]["text"] == "summary"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
def test_capabilities_returns_all_true(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    caps = adapter.capabilities()

    assert isinstance(caps, AdapterCapabilities)
    assert caps.supports_stm is True
    assert caps.supports_ltm is True
    assert caps.supports_dialog_graph is True
    assert caps.supports_artifact_ingest is True
    assert caps.supports_recall is True
    assert caps.supports_promote is True


@patch("graphknows.memory.adapter.LTMPromotionPipeline")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
@pytest.mark.asyncio
async def test_promote_calls_session_flusher(
    mock_dialog_writer,
    mock_ingestor,
    mock_pipeline,
):
    adapter = _make_adapter()
    mock_pipeline.return_value.promote = AsyncMock(
        return_value=MagicMock(
            nodes_written=10,
            edges_written=5,
            chunks_promoted=3,
            turns_promoted=2,
            topics_created=1,
            merged_entities=0,
            elapsed_s=1.5,
            errors=[],
        )
    )
    options = PromoteOptions(
        session_id="s1",
        run_gds=True,
        run_topics=True,
    )
    result = await adapter.promote(options)

    mock_pipeline.return_value.promote.assert_awaited_once()
    (request,), _ = mock_pipeline.return_value.promote.call_args
    assert request.session_id == "s1"
    assert request.run_gds is True
    assert result["session_id"] == "s1"
    assert result["run_gds"] is True
    assert result["run_topics"] is True
    assert result["nodes_written"] == 10
    assert result["edges_written"] == 5
    assert result["chunks_promoted"] == 3
    assert result["turns_promoted"] == 2
    assert result["topics_created"] == 1
    assert result["merged_entities"] == 0
    assert result["elapsed_s"] == 1.5
    assert result["errors"] == []


@patch("graphknows.memory.adapter.LTMPromotionPipeline")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogCapture")
@pytest.mark.asyncio
async def test_promote_always_summarizes_topics(
    mock_dialog_writer,
    mock_ingestor,
    mock_pipeline,
):
    adapter = _make_adapter()
    mock_pipeline.return_value.promote = AsyncMock(
        return_value=MagicMock(
            nodes_written=1,
            edges_written=1,
            chunks_promoted=1,
            turns_promoted=0,
            topics_created=1,
            merged_entities=0,
            elapsed_s=0.1,
            errors=[],
        )
    )
    options = PromoteOptions(
        session_id="s1",
        run_gds=False,
        run_topics=False,
    )
    result = await adapter.promote(options)

    (request,), _ = mock_pipeline.return_value.promote.call_args
    assert request.run_gds is False
    assert result["run_topics"] is True


# ===========================================================================
# Adapter conformance suite
# ===========================================================================


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.adapter import GraphknowsAdapter
from graphknows.memory.adapter_types import AdapterCapabilities, RecallRequest
from graphknows.memory.models.recall import RecallResult
from tests.fixtures.synthetic_host_session import (
    ARTIFACTS,
    PROMOTE_OPTIONS,
    RECALL_REQUEST,
    SESSION_ID,
    TURNS,
)


def _make_adapter():
    """Return a GraphknowsAdapter with all stores mocked."""
    return GraphknowsAdapter(
        mg_store=MagicMock(),
        pg_store=MagicMock(),
        neo_store=MagicMock(),
        extractor=MagicMock(),
        embed_fn=None,
    )


class TestAdapterLifecycle:
    """Conformance tests for full adapter lifecycle sequences."""

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_full_session_write_sequence(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        assert adapter._active_session_id == SESSION_ID

        for turn in TURNS:
            adapter.write_turn(turn)
        assert mock_dw_cls.return_value.write_turn.call_count == len(TURNS)

        for artifact in ARTIFACTS:
            adapter.ingest_artifact(artifact, SESSION_ID)
        assert mock_ing_cls.return_value.ingest_text.call_count == len(ARTIFACTS)

    @patch("graphknows.memory.stm.recall.recall_short_term")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_recall_after_write(self, mock_dw_cls, mock_ing_cls, mock_recall):
        mock_recall.return_value = [
            RecallResult(
                text="EU AI Act fact",
                source_type="chunk",
                scope="short_term",
                score=0.9,
                metadata={"memory_class": "fact"},
            )
        ]
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        for turn in TURNS:
            adapter.write_turn(turn)

        result = adapter.recall(RECALL_REQUEST)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "EU AI Act fact"
        assert "source_type" in result[0]

    @patch("graphknows.memory.adapter.LTMPromotionPipeline")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    @pytest.mark.asyncio
    async def test_promote_after_write(self, mock_dw_cls, mock_ing_cls, mock_pipeline_cls):
        mock_promote_result = MagicMock()
        mock_promote_result.__dict__ = {"session_id": SESSION_ID}
        mock_pipeline_cls.return_value.promote = AsyncMock(return_value=mock_promote_result)

        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.write_turn(TURNS[0])

        result = await adapter.promote(PROMOTE_OPTIONS)
        mock_pipeline_cls.return_value.promote.assert_called_once()
        (request,), _ = mock_pipeline_cls.return_value.promote.call_args
        assert request.session_id == SESSION_ID
        assert isinstance(result, dict)

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_capabilities_contract(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.supports_stm is True
        assert caps.supports_ltm is True
        assert caps.supports_dialog_graph is True
        assert caps.supports_artifact_ingest is True
        assert caps.supports_recall is True
        assert caps.supports_promote is True

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_artifact_metadata_includes_memory_class(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.ingest_artifact(ARTIFACTS[0], SESSION_ID)

        call_kwargs = mock_ing_cls.return_value.ingest_text.call_args[1]
        assert call_kwargs["metadata"]["memory_class"] == "raw_evidence"

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_open_session_idempotent(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.open_session(SESSION_ID)  # second call — no exception
        assert adapter._active_session_id == SESSION_ID

    @patch("graphknows.memory.stm.recall.recall_short_term")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_recall_include_raw_false_filters(self, mock_dw_cls, mock_ing_cls, mock_recall):
        mock_recall.return_value = [
            RecallResult(
                text="raw hit",
                source_type="chunk",
                scope="short_term",
                score=0.9,
                provenance={"memory_class": "raw_evidence"},
            ),
            RecallResult(
                text="fact hit",
                source_type="chunk",
                scope="short_term",
                score=0.8,
                provenance={"memory_class": "fact"},
            ),
        ]
        adapter = _make_adapter()
        req = RecallRequest(query="test", session_id=SESSION_ID, top_k=5, include_raw=False)
        result = adapter.recall(req)
        texts = [r["text"] for r in result]
        assert "raw hit" not in texts
        assert "fact hit" in texts
