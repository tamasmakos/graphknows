"""Unit tests for ADAPT-001: AgentMemoryAdapter protocol + GraphknowsMemoryAdapter."""

from __future__ import annotations

from typing import Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.adapter import (
    AgentMemoryAdapter,
    FlushResult,
    GraphknowsMemoryAdapter,
    MemoryStats,
    RecallHit,
)
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.models.ingest_options import IngestOptions
from graphknows.memory.models.scope import MemoryScope

# ---------------------------------------------------------------------------
# DTO tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_result_dataclass():
    r = IngestResult(session_id="s1", file_id="f1", title="", chunk_count=3, entity_count=5, elapsed_s=1.2)
    assert r.file_id == "f1"
    assert r.chunk_count == 3
    assert r.entity_count == 5
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
    internal_result = IngestResult(
        session_id="r:s1", file_id="dialog-r:s1", title="",
        chunk_count=2, entity_count=5, elapsed_s=1.2,
    )

    mock_svc = MagicMock()
    mock_svc.ingest_messages = MagicMock(return_value=internal_result)

    with (
        patch(
            "graphknows.memory.extractor.GLiNER2EntityExtractor",
            return_value=MagicMock(),
        ),
        patch(
            "graphknows.memory.adapter.STMService",
            return_value=mock_svc,
        ),
        patch(
            "graphknows.memory.adapter.asyncio.to_thread",
            new=_sync_to_thread,
        ),
        patch("graphknows.memory.stm._async_bridge._run"),
    ):
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        options = IngestOptions(scope=MemoryScope(run_id="s1"))
        result = await adapter.ingest(
            messages=[{"role": "user", "content": "hello world"}],
            options=options,
        )

    assert isinstance(result, IngestResult)
    assert result.file_id == "dialog-r:s1"
    assert result.chunk_count == 2
    assert result.entity_count == 5


# ---------------------------------------------------------------------------
# GraphknowsMemoryAdapter — recall
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_stm(mock_container):
    from graphknows.memory.query.context import RetrievalContext

    fake_ctx = RetrievalContext(
        query="test",
        session_id="s1",
        fused_results=[
            {"chunk_id": "c1", "doc_id": "", "text": "stm hit", "metadata": {},
             "rrf_score": 0.9, "sources": "stm_graph", "rank": 1, "entities": ["e1"]}
        ],
        search_type="DET",
    )

    with patch("graphknows.memory.query.retriever.DETRetriever") as MockDET:
        MockDET.return_value.retrieve = AsyncMock(return_value=fake_ctx)
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(query="test", scope="s1", top_k=5, recall_scope="stm")

    assert len(result) == 1
    assert isinstance(result[0], RecallHit)
    assert result[0].score == 0.9
    assert result[0].text == "stm hit"
    assert result[0].entities == ["e1"]
    assert result[0].session_id == "s1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_ltm(mock_container):
    from graphknows.memory.query.context import RetrievalContext

    fake_ctx = RetrievalContext(
        query="test",
        session_id="s1",
        fused_results=[
            {"chunk_id": "c2", "doc_id": "", "text": "ltm hit", "metadata": {},
             "rrf_score": 0.8, "sources": "ltm_graph", "rank": 1, "entities": []}
        ],
        search_type="DET",
    )

    with patch("graphknows.memory.query.retriever.DETRetriever") as MockDET:
        MockDET.return_value.retrieve = AsyncMock(return_value=fake_ctx)
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(query="test", scope="s1", top_k=5, recall_scope="ltm")

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].text == "ltm hit"
    assert result[0].source_type == "ltm_graph"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_scope_both(mock_container):
    from graphknows.memory.query.context import RetrievalContext

    fake_ctx = RetrievalContext(
        query="test",
        session_id="s1",
        fused_results=[
            {"chunk_id": "c1", "doc_id": "", "text": "stm hit", "metadata": {},
             "rrf_score": 0.9, "sources": "stm_graph", "rank": 1, "entities": ["e1"]},
            {"chunk_id": "c2", "doc_id": "", "text": "ltm hit", "metadata": {},
             "rrf_score": 0.7, "sources": "ltm_graph", "rank": 2, "entities": []},
        ],
        search_type="DET",
    )

    with patch("graphknows.memory.query.retriever.DETRetriever") as MockDET:
        MockDET.return_value.retrieve = AsyncMock(return_value=fake_ctx)
        adapter = GraphknowsMemoryAdapter(container=mock_container)
        result = await adapter.recall(query="test", scope="s1", top_k=5, recall_scope="both")

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
        result = await adapter.flush(scope="s1")

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


def _make_adapter_full():
    mg = MagicMock()
    pg = MagicMock()
    neo = MagicMock()
    extractor = MagicMock()
    embed_fn = MagicMock(return_value=[0.1] * 10)
    return GraphknowsAdapter(mg, pg, neo, extractor, embed_fn)


@patch("graphknows.memory.adapter.STMService")
@patch("graphknows.memory.adapter.DialogCapture")
def test_open_session_sets_active_session_id(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter_full()
    adapter.open_session("s1")
    assert adapter._active_session_id == "s1"


@patch("graphknows.memory.adapter.STMService")
@patch("graphknows.memory.adapter.DialogCapture")
def test_ingest_artifact_delegates_to_ingestor(
    mock_dialog_writer,
    mock_stm_cls,
):
    adapter = _make_adapter_full()
    artifact = ArtifactRecord(
        artifact_id="a1",
        path="/tmp/test.py",
        content="print('hello')",
        mime_type="text/x-python",
        memory_class=MemoryClass.raw_evidence,
    )
    adapter.ingest_artifact(artifact, "s1")

    mock_stm_cls.return_value.ingest.assert_called_once()
    args, kwargs = mock_stm_cls.return_value.ingest.call_args
    assert args[0] == "print('hello')"
    assert kwargs.get("session_id") == "s1"
    assert kwargs.get("title") == "/tmp/test.py"
    assert kwargs.get("source_hint") == "text/x-python"
    metadata = kwargs.get("metadata", {})
    assert metadata.get("memory_class") == "raw_evidence"
    assert metadata.get("artifact_id") == "a1"


def _sync_run_for_tests(coro):
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_recall_returns_hits_as_dicts():
    adapter = _make_adapter()

    async def stm_chunks(*a, **k):
        return [{"id": "c1", "text": "hello", "cosine_score": 0.9,
                 "file_id": "f1", "heading_path": "", "metadata": {"memory_class": "chunk"}}]

    async def empty(*a, **k):
        return []

    adapter.pg_store.search_stm_chunks = stm_chunks
    adapter.pg_store.search_stm_turns = empty
    adapter.mg_store.query_entity_chunks = empty

    request = RecallRequest(query="test query here", session_id="s1", top_k=5)
    with patch("graphknows.memory.stm._async_bridge._run", _sync_run_for_tests):
        result = adapter.recall(request)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "hello"
    assert result[0]["source_type"] == "chunk"
    assert result[0]["scope"] == "short_term"


def test_recall_filters_raw_evidence_when_include_raw_false():
    adapter = _make_adapter()

    async def stm_chunks(*a, **k):
        return [
            {"id": "c1", "text": "raw", "cosine_score": 0.9,
             "file_id": "", "heading_path": "", "metadata": {"memory_class": "raw_evidence"}},
            {"id": "c2", "text": "summary", "cosine_score": 0.8,
             "file_id": "", "heading_path": "", "metadata": {"memory_class": "summary"}},
        ]

    async def empty(*a, **k):
        return []

    adapter.pg_store.search_stm_chunks = stm_chunks
    adapter.pg_store.search_stm_turns = empty
    adapter.mg_store.query_entity_chunks = empty

    request = RecallRequest(query="test query here", session_id="s1", top_k=5, include_raw=False)
    with patch("graphknows.memory.stm._async_bridge._run", _sync_run_for_tests):
        result = adapter.recall(request)

    assert len(result) == 1
    assert result[0]["text"] == "summary"


@patch("graphknows.memory.adapter.STMService")
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
@patch("graphknows.memory.adapter.STMService")
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
@patch("graphknows.memory.adapter.STMService")
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

    @patch("graphknows.memory.adapter.STMService")
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
        assert mock_ing_cls.return_value.ingest.call_count == len(ARTIFACTS)

    @patch("graphknows.memory.adapter.DialogCapture")
    def test_recall_after_write(self, mock_dw_cls):
        import asyncio

        def _sync_run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        fake_row = {
            "id": "c1", "text": "EU AI Act fact",
            "metadata": {"memory_class": "fact"},
            "file_id": "", "cosine_score": 0.9, "heading_path": "",
        }
        pg = MagicMock()
        pg.search_stm_chunks = AsyncMock(return_value=[fake_row])
        pg.search_stm_turns = AsyncMock(return_value=[])
        mg = MagicMock()
        mg.query_entity_chunks = AsyncMock(return_value=[])

        # Use a real STMService backed by mock stores — STMService must NOT be
        # patched here because recall() now delegates to _stm_service.recall().
        adapter = GraphknowsAdapter(
            mg_store=mg, pg_store=pg, neo_store=MagicMock(),
            extractor=MagicMock(), embed_fn=lambda q: [0.0] * 1536,
        )
        adapter.open_session(SESSION_ID)
        for turn in TURNS:
            adapter.write_turn(turn)

        with patch("graphknows.memory.stm._async_bridge._run", side_effect=_sync_run):
            result = adapter.recall(RECALL_REQUEST)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "EU AI Act fact"
        assert "source_type" in result[0]

    @patch("graphknows.memory.adapter.LTMPromotionPipeline")
    @patch("graphknows.memory.adapter.STMService")
    @patch("graphknows.memory.adapter.DialogCapture")
    @pytest.mark.asyncio
    async def test_promote_after_write(self, mock_dw_cls, mock_ing_cls, mock_pipeline_cls):
        from graphknows.memory.models.promotion import PromotionRecord

        mock_promote_result = PromotionRecord(session_id=SESSION_ID)
        mock_pipeline_cls.return_value.promote = AsyncMock(return_value=mock_promote_result)

        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.write_turn(TURNS[0])

        result = await adapter.promote(PROMOTE_OPTIONS)
        mock_pipeline_cls.return_value.promote.assert_called_once()
        (request,), _ = mock_pipeline_cls.return_value.promote.call_args
        assert request.session_id == SESSION_ID
        assert isinstance(result, dict)

    @patch("graphknows.memory.adapter.STMService")
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

    @patch("graphknows.memory.adapter.STMService")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_artifact_metadata_includes_memory_class(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.ingest_artifact(ARTIFACTS[0], SESSION_ID)

        call_kwargs = mock_ing_cls.return_value.ingest.call_args[1]
        assert call_kwargs["metadata"]["memory_class"] == "raw_evidence"

    @patch("graphknows.memory.adapter.STMService")
    @patch("graphknows.memory.adapter.DialogCapture")
    def test_open_session_idempotent(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.open_session(SESSION_ID)  # second call — no exception
        assert adapter._active_session_id == SESSION_ID

    @patch("graphknows.memory.adapter.DialogCapture")
    def test_recall_include_raw_false_filters(self, mock_dw_cls):
        import asyncio

        def _sync_run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        rows = [
            {"id": "c1", "text": "raw hit", "metadata": {"memory_class": "raw_evidence"},
             "file_id": "", "cosine_score": 0.9, "heading_path": ""},
            {"id": "c2", "text": "fact hit", "metadata": {"memory_class": "fact"},
             "file_id": "", "cosine_score": 0.8, "heading_path": ""},
        ]
        pg = MagicMock()
        pg.search_stm_chunks = AsyncMock(return_value=rows)
        pg.search_stm_turns = AsyncMock(return_value=[])
        mg = MagicMock()
        mg.query_entity_chunks = AsyncMock(return_value=[])

        adapter = GraphknowsAdapter(
            mg_store=mg, pg_store=pg, neo_store=MagicMock(),
            extractor=MagicMock(), embed_fn=lambda q: [0.0] * 1536,
        )
        req = RecallRequest(query="test", session_id=SESSION_ID, top_k=5, include_raw=False)
        with patch("graphknows.memory.stm._async_bridge._run", side_effect=_sync_run):
            result = adapter.recall(req)
        texts = [r["text"] for r in result]
        assert "raw hit" not in texts
        assert "fact hit" in texts
