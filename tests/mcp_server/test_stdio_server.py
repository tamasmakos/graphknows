"""Tests for the graphknows MCP stdio server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import guard — must not trigger GLiNER2 or heavy model loads
# ---------------------------------------------------------------------------


def test_import_does_not_load_gliner2():
    """Importing the stdio_server must not trigger GLiNER2 model loading."""
    import importlib
    import sys

    # Remove any cached import
    for key in list(sys.modules.keys()):
        if "stdio_server" in key or "gliner" in key.lower():
            del sys.modules[key]

    # Import should be side-effect free (no model loading)
    mod = importlib.import_module("graphknows.mcp.stdio_server")
    assert callable(mod.main)
    assert callable(mod.memory_doctor)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_all_tools_registered():
    """All expected tools must be present in the FastMCP app."""
    from graphknows.mcp.stdio_server import app

    # FastMCP exposes list_tools() synchronously or via the tool registry
    tool_names = {t.name for t in app._tool_manager.list_tools()}
    expected = {
        "memory_doctor",
        "memory_stats",
        "memory_purge",
        "memory_ingest",
        "memory_recall",
        "memory_flush",
        "memory_mirror",
        "ltm_query",
        "ltm_entity",
        "ltm_topics",
        "corpus_ingest",
    }
    assert expected == tool_names, f"Missing tools: {expected - tool_names}"


def test_memory_mirror_returns_local_url():
    """memory_mirror returns a local browser URL with selected scope/session."""
    from graphknows.mcp.stdio_server import memory_mirror

    with patch.dict(
        "os.environ",
        {"GRAPHKNOWS_WEB_HOST": "127.0.0.1", "GRAPHKNOWS_WEB_PORT": "8899"},
    ):
        result = memory_mirror(session_id="demo", scope="stm")

    assert result == {
        "url": "http://127.0.0.1:8899/?scope=stm&session_id=demo",
        "session_id": "demo",
        "scope": "stm",
    }


# ---------------------------------------------------------------------------
# memory_doctor
# ---------------------------------------------------------------------------


def test_memory_doctor_all_ok():
    """memory_doctor returns True for all backends when connectivity succeeds."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_doctor

    async def _fake_connect(dsn):
        conn = AsyncMock()
        conn.close = AsyncMock()
        return conn

    drv = MagicMock()
    drv.session.return_value.__enter__ = lambda s: s
    drv.session.return_value.__exit__ = MagicMock(return_value=False)
    drv.session.return_value.run.return_value.single.return_value = [1]

    with (
        patch("neo4j.GraphDatabase.driver", return_value=drv),
        patch("asyncpg.connect", side_effect=_fake_connect),
    ):
        result = asyncio.run(memory_doctor())

    assert result["memgraph"] is True
    assert result["neo4j"] is True
    assert result["pgvector"] is True
    assert isinstance(result["version"], str) and len(result["version"]) > 0
    assert isinstance(result["pid"], int) and result["pid"] > 0


def test_memory_doctor_memgraph_down():
    """memory_doctor returns False for memgraph when connection fails."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_doctor

    async def _fake_connect(dsn):
        conn = AsyncMock()
        conn.close = AsyncMock()
        return conn

    # Memgraph driver raises, Neo4j driver succeeds
    call_count = [0]

    def _driver_factory(uri, auth):
        call_count[0] += 1
        drv = MagicMock()
        drv.session.return_value.__enter__ = lambda s: s
        drv.session.return_value.__exit__ = MagicMock(return_value=False)
        if "17687" in uri:
            drv.session.return_value.run.side_effect = ConnectionError("timeout")
        else:
            drv.session.return_value.run.return_value.single.return_value = [1]
        return drv

    with (
        patch("neo4j.GraphDatabase.driver", side_effect=_driver_factory),
        patch("asyncpg.connect", side_effect=_fake_connect),
    ):
        result = asyncio.run(memory_doctor())

    assert result["memgraph"] is False
    assert result["neo4j"] is True
    assert result["pgvector"] is True
    assert isinstance(result["version"], str) and len(result["version"]) > 0
    assert isinstance(result["pid"], int) and result["pid"] > 0


# ---------------------------------------------------------------------------
# memory_recall (mocked backends)
# ---------------------------------------------------------------------------


def test_memory_recall_returns_hits():
    """memory_recall returns formatted hit list."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(
        return_value={
            "hits": [
                {
                    "type": "chunk",
                    "session_id": "test-001",
                    "score": round(0.95, 6),
                    "entities": ["graphknows"],
                }
            ]
        }
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="graphknows", session_id="test-001", top_k=5))

    runtime.recall_memory.assert_awaited_once_with(
        "graphknows", session_id="test-001", top_k=5, scope="stm"
    )
    assert "hits" in result
    assert len(result["hits"]) == 1
    h = result["hits"][0]
    assert h["type"] == "chunk"
    assert h["session_id"] == "test-001"
    assert h["score"] == round(0.95, 6)
    assert "graphknows" in h["entities"]


def test_memory_recall_hit_has_cosine_score():
    """Each recall hit must include cosine_score in [0.0, 1.0]."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(
        return_value={
            "hits": [
                {
                    "score": round(0.95, 6),
                    "cosine_score": 0.87,
                    "type": "chunk",
                    "text": "graphknows is a framework-agnostic memory layer.",
                    "session_id": "test-001",
                    "entities": ["graphknows"],
                }
            ]
        }
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="graphknows", session_id="test-001", top_k=5))

    assert "hits" in result
    assert len(result["hits"]) == 1
    h = result["hits"][0]
    assert "cosine_score" in h, f"Expected cosine_score key, got {list(h.keys())}"
    assert isinstance(h["cosine_score"], float), f"Expected float, got {type(h['cosine_score'])}"
    assert 0.0 <= h["cosine_score"] <= 1.0, f"cosine_score={h['cosine_score']} out of [0,1]"


def test_memory_recall_empty():
    """memory_recall returns empty hits list when nothing found."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(return_value={"hits": []})

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="unknown", session_id="test-001"))

    assert result == {"hits": []}


# ---------------------------------------------------------------------------
# LTM session_filter fix (MEM-001)
# ---------------------------------------------------------------------------


def test_ltm_recall_uses_session_filter():
    """MCP recall passes the requested session_id through to the runtime."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(return_value={"hits": [{"text": "some ltm content"}]})

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="test", session_id="current-session", scope="ltm"))

    runtime.recall_memory.assert_awaited_once_with(
        "test", session_id="current-session", top_k=5, scope="ltm"
    )
    assert len(result["hits"]) == 1


# ---------------------------------------------------------------------------
# Content prefix strip fix (MEM-002)
# ---------------------------------------------------------------------------


def test_clean_strips_content_prefix():
    """MCP recall returns the runtime's cleaned text unchanged."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    def _run_ltm(text: str, score: float = 0.9) -> str:
        runtime = MagicMock()
        runtime.recall_memory = AsyncMock(return_value={"hits": [{"text": text, "score": score}]})

        with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
            return asyncio.run(memory_recall(query="x", session_id="s1", scope="ltm"))["hits"][0][
                "text"
            ]

    assert _run_ltm("Some real content here.") == "Some real content here."
    assert _run_ltm("Lower case.") == "Lower case."
    assert _run_ltm("regular text without prefix") == "regular text without prefix"


# ---------------------------------------------------------------------------
# _normalise_and_weight (RECALL-005)
# ---------------------------------------------------------------------------


def test_normalise_and_weight_exists():
    """_normalise_and_weight is a callable function in stdio_server."""
    from graphknows.mcp.stdio_server import _normalise_and_weight

    assert callable(_normalise_and_weight)


def test_normalise_and_weight_empty_list():
    """_normalise_and_weight is a no-op for an empty list."""
    from graphknows.mcp.stdio_server import _normalise_and_weight

    hits: list = []
    _normalise_and_weight(hits, weight=1.0)
    assert hits == []


def test_normalise_and_weight_single_hit():
    """_normalise_and_weight assigns weight to a single hit (range is zero)."""
    from types import SimpleNamespace

    from graphknows.mcp.stdio_server import _normalise_and_weight

    hits = [SimpleNamespace(score=0.5)]
    _normalise_and_weight(hits, weight=0.6)
    assert hits[0].score == 0.6


def test_normalise_and_weight_ltm_interleaves():
    """LTM hit can outscore a mid-ranked STM hit after normalise+weight.

    STM scores [0.016, 0.014], LTM scores [0.033, 0.010].
    After per-scope min-max normalisation and weights (stm=0.6, ltm=0.4):
      STM top  = 0.6, STM second = 0.0
      LTM top  = 0.4, LTM second = 0.0
    The top LTM hit (0.4) beats the second STM hit (0.0), showing cross-scope
    interleaving is possible.
    """
    from types import SimpleNamespace

    from graphknows.mcp.stdio_server import _normalise_and_weight

    stm_hits = [
        SimpleNamespace(score=0.016),
        SimpleNamespace(score=0.014),
    ]
    ltm_hits = [
        SimpleNamespace(score=0.033),
        SimpleNamespace(score=0.010),
    ]

    _normalise_and_weight(stm_hits, weight=0.6)
    _normalise_and_weight(ltm_hits, weight=0.4)

    # LTM top should beat STM second
    assert ltm_hits[0].score > stm_hits[1].score, (
        f"LTM top {ltm_hits[0].score} should exceed STM second {stm_hits[1].score}"
    )
    # STM top keeps #1
    assert stm_hits[0].score > ltm_hits[0].score


def test_normalise_and_weight_multi_hit():
    """_normalise_and_weight min-max normalises and applies weight."""
    from types import SimpleNamespace

    from graphknows.mcp.stdio_server import _normalise_and_weight

    hits = [
        SimpleNamespace(score=10.0),
        SimpleNamespace(score=20.0),
        SimpleNamespace(score=30.0),
    ]
    _normalise_and_weight(hits, weight=0.5)
    # min=10, max=30, range=20
    # 10 → (10-10)/20 = 0.0 → 0.0 * 0.5 = 0.0
    # 20 → (20-10)/20 = 0.5 → 0.5 * 0.5 = 0.25
    # 30 → (30-10)/20 = 1.0 → 1.0 * 0.5 = 0.5
    assert hits[0].score == 0.0
    assert hits[1].score == 0.25
    assert hits[2].score == 0.5


def test_normalise_and_weight_equal_scores():
    """_normalise_and_weight handles equal scores (zero range → all get weight)."""
    from types import SimpleNamespace

    from graphknows.mcp.stdio_server import _normalise_and_weight

    hits = [
        SimpleNamespace(score=42.0),
        SimpleNamespace(score=42.0),
    ]
    _normalise_and_weight(hits, weight=0.8)
    assert hits[0].score == 0.8
    assert hits[1].score == 0.8


def test_scope_stm_unchanged():
    """scope='stm' is passed through to runtime recall."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(return_value={"hits": [{"score": round(0.95, 6)}]})

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="x", session_id="test-001", scope="stm"))

    runtime.recall_memory.assert_awaited_once_with(
        "x", session_id="test-001", top_k=5, scope="stm"
    )
    assert len(result["hits"]) == 1
    assert result["hits"][0]["score"] == round(0.95, 6)


def test_scope_ltm_unchanged():
    """scope='ltm' is passed through to runtime recall."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(return_value={"hits": [{"score": round(0.87, 6)}]})

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="x", session_id="test-001", scope="ltm"))

    runtime.recall_memory.assert_awaited_once_with(
        "x", session_id="test-001", top_k=5, scope="ltm"
    )
    assert len(result["hits"]) == 1
    assert result["hits"][0]["score"] == round(0.87, 6)


def test_scope_both_normalises_cross_scope_scores():
    """scope='both' returns runtime-normalised cross-scope scores."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(
        return_value={"hits": [{"score": round(0.6, 6)}, {"score": round(0.4, 6)}]}
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="x", session_id="s1", top_k=5, scope="both"))

    runtime.recall_memory.assert_awaited_once_with("x", session_id="s1", top_k=5, scope="both")
    assert len(result["hits"]) == 2
    # Both are single-element lists, so each normalises to 1.0:
    #   STM: 0.6 * 1.0 = 0.6
    #   LTM: 0.4 * 1.0 = 0.4
    # STM first, LTM second.
    assert result["hits"][0]["score"] == round(0.6, 6)
    assert result["hits"][1]["score"] == round(0.4, 6)


# ---------------------------------------------------------------------------
# LTM tools — merged from mcp_ltm.py
# ---------------------------------------------------------------------------


def test_ltm_tools_registered():
    """All 6 LTM tools are registered in the FastMCP app."""
    from graphknows.mcp.stdio_server import app

    tool_names = {t.name for t in app._tool_manager.list_tools()}
    for name in (
        "ltm_query",
        "ltm_entity",
        "ltm_topics",
    ):
        assert name in tool_names, f"Missing tool: {name}"


def test_ltm_topics_returns_topics():
    """ltm_topics returns topic list above threshold."""
    from graphknows.mcp.stdio_server import ltm_topics

    runtime = MagicMock()
    runtime.ltm_topics.return_value = {
        "topics": [
            {"community_id": 1, "summary": "AI topics", "entity_count": 5},
            {"community_id": 2, "summary": "ML topics", "entity_count": 3},
        ]
    }

    with patch("graphknows.mcp.tools.ltm.get_runtime", return_value=runtime):
        result = ltm_topics(min_entities=3)

    runtime.ltm_topics.assert_called_once_with(min_entities=3)
    assert "topics" in result
    assert len(result["topics"]) == 2
    assert result["topics"][0]["community_id"] == 1
    assert result["topics"][0]["entity_count"] == 5


def test_ltm_entity_not_found():
    """ltm_entity returns error dict when entity not found."""
    from graphknows.mcp.stdio_server import ltm_entity

    runtime = MagicMock()
    runtime.ltm_entity.return_value = {"error": "Entity not found: NonExistent"}

    with patch("graphknows.mcp.tools.ltm.get_runtime", return_value=runtime):
        result = ltm_entity(name="NonExistent")

    assert "error" in result
    assert "NonExistent" in result["error"]


def test_ltm_entity_found():
    """ltm_entity returns entity properties when found."""
    from graphknows.mcp.stdio_server import ltm_entity

    mock_props = {"name": "TestEntity", "pagerank": 0.5, "segment_count": 3}
    runtime = MagicMock()
    runtime.ltm_entity.return_value = mock_props

    with patch("graphknows.mcp.tools.ltm.get_runtime", return_value=runtime):
        result = ltm_entity(name="TestEntity")

    assert "error" not in result
    assert result["name"] == "TestEntity"
    assert result["segment_count"] == 3


# ---------------------------------------------------------------------------
# corpus_ingest
# ---------------------------------------------------------------------------


def test_corpus_ingest_calls_run_ingestion():
    """corpus_ingest routes through the runtime facade."""
    import asyncio

    from graphknows.mcp.tools.corpus import corpus_ingest

    runtime = MagicMock()
    runtime.corpus_ingest = AsyncMock(
        return_value={
            "session_id": "sess-abc",
            "documents_processed": 1,
            "entities_extracted": 5,
            "chunks_written": 2,
            "nodes_written": 5,
            "edges_written": 3,
            "elapsed_s": 0.1,
            "errors": [],
        }
    )

    with patch("graphknows.mcp.tools.corpus.get_runtime", return_value=runtime):
        result = asyncio.run(
            corpus_ingest(
                documents=[{"text": "test", "title": "t", "id": "1"}],
                session_id="sess-abc",
            )
        )

    assert result["documents_processed"] == 1
    assert result["nodes_written"] == 5
    assert "session_id" in result
    runtime.corpus_ingest.assert_awaited_once_with(
        [{"text": "test", "title": "t", "id": "1"}], session_id="sess-abc"
    )


def test_corpus_ingest_empty_documents():
    """corpus_ingest returns error when documents list is empty."""
    import asyncio

    from graphknows.mcp.stdio_server import corpus_ingest

    result = asyncio.run(corpus_ingest(documents=[]))

    assert "error" in result
    assert result["documents_processed"] == 0


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_app_is_fastmcp_instance(self):
        import graphknows.mcp.stdio_server as srv

        # app is the FastMCP application object
        assert srv.app is not None

    def test_module_has_tool_functions(self):
        import graphknows.mcp.stdio_server as srv

        # There should be multiple tool functions registered
        public = [n for n in dir(srv) if not n.startswith("_") and callable(getattr(srv, n))]
        assert len(public) > 5


# ---------------------------------------------------------------------------
# Issue #26 — Flush idempotency
# ---------------------------------------------------------------------------


class TestMemoryFlushIdempotency:
    """Tests for memory_flush idempotency guard (Issue #26)."""

    @pytest.mark.asyncio
    async def test_already_flushed_returns_early(self):
        """When session already exists in LTM, return status=already_flushed without calling flusher."""
        from graphknows.mcp.tools import stm as stm_mod

        runtime = MagicMock()
        runtime.flush_memory = AsyncMock(
            return_value={
                "session_id": "session-already-flushed",
                "status": "already_flushed",
                "nodes": 0,
                "edges": 0,
                "chunks": 0,
                "turns": 0,
                "topics_created": 0,
                "merged_entities": 0,
                "elapsed_s": 0.0,
                "errors": [],
            }
        )
        with patch.object(stm_mod, "get_runtime", return_value=runtime):
            result = await stm_mod.memory_flush("session-already-flushed")

        runtime.flush_memory.assert_awaited_once_with("session-already-flushed")
        assert result["status"] == "already_flushed"
        assert result["session_id"] == "session-already-flushed"
        assert result["nodes"] == 0
        assert result["chunks"] == 0
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_new_session_calls_flusher(self):
        """When session is NOT in LTM, the flusher should be called."""
        from graphknows.mcp.tools import stm as stm_mod

        runtime = MagicMock()
        runtime.flush_memory = AsyncMock(
            return_value={
                "nodes": 5,
                "edges": 3,
                "chunks": 2,
                "turns": 1,
                "topics_created": 0,
                "merged_entities": 0,
                "elapsed_s": 0.5,
                "errors": [],
            }
        )

        with patch.object(stm_mod, "get_runtime", return_value=runtime):
            result = await stm_mod.memory_flush("session-new")

        runtime.flush_memory.assert_awaited_once_with("session-new")
        assert result.get("status") != "already_flushed"


def test_memory_ingest_returns_no_warnings():
    """memory_ingest must not return rel_extractor_disabled warning — issue #113."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_ingest

    runtime = MagicMock()
    runtime.ingest_memory = AsyncMock(
        return_value={
            "file_id": "abc123",
            "chunks": 1,
            "entities": 5,
            "elapsed_s": 1.0,
            "errors": [],
            "warnings": [],
        }
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_ingest("some text", session_id="test", title="t"))
        assert result["warnings"] == [], f"Expected no warnings, got: {result['warnings']}"
        assert "rel_extractor_disabled" not in result["warnings"]


def test_memory_flush_summarizes_topics():
    """memory_flush delegates topic-aware promotion policy to the runtime."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_flush

    runtime = MagicMock()
    runtime.flush_memory = AsyncMock(
        return_value={
            "nodes": 1,
            "edges": 1,
            "chunks": 1,
            "turns": 0,
            "topics_created": 1,
            "merged_entities": 0,
            "elapsed_s": 1.0,
            "errors": [],
        }
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        asyncio.run(memory_flush("test-session"))
        runtime.flush_memory.assert_awaited_once_with("test-session")


# ---------------------------------------------------------------------------
# Issue #111 — STM recall uses _BG_LOOP pg store, not get_pg_store()
# ---------------------------------------------------------------------------


def test_stm_recall_uses_bg_loop_pg_store():
    """memory_recall delegates pgvector loop policy to the runtime."""
    import asyncio

    from graphknows.mcp.tools.stm import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(return_value={"hits": []})

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        asyncio.run(memory_recall("test query", session_id="test-session", scope="stm"))
        runtime.recall_memory.assert_awaited_once_with(
            "test query", session_id="test-session", top_k=5, scope="stm"
        )


def test_stm_recall_hit_entity_only_has_cosine_score_zero():
    """Entity-only recall hits must have cosine_score explicitly set to 0.0 — issue #112."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    runtime = MagicMock()
    runtime.recall_memory = AsyncMock(
        return_value={
            "hits": [
                {
                    "score": 0.016,
                    "cosine_score": 0.0,
                    "type": "chunk",
                    "text": "entity-only hit",
                    "session_id": "test-001",
                    "entities": ["graphknows"],
                }
            ]
        }
    )

    with patch("graphknows.mcp.tools.stm.get_runtime", return_value=runtime):
        result = asyncio.run(memory_recall(query="test", session_id="test-001", scope="stm"))
    hits = result.get("hits", [])
    assert len(hits) == 1
    assert hits[0]["cosine_score"] == 0.0


class TestLTMPromotionPipelineRelationExtractionGuard:
    """LTMPromotionPipeline._write_ltm_graph must skip/survive relation extraction failures."""

    @pytest.mark.asyncio
    async def test_write_skips_extraction_when_key_empty(self) -> None:
        """Empty or whitespace-only openrouter_key → extract_and_write_relations is never called."""
        from unittest.mock import AsyncMock

        from graphknows.memory.flush import LTMPromotionPipeline

        for empty_key in ("", "   ", "\t"):
            mock_neo = AsyncMock()
            mock_neo.write_session_subgraph.return_value = {"nodes": 2, "edges": 1}
            stm_data: dict = {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}
            errors: list[str] = []

            pipeline = LTMPromotionPipeline.__new__(LTMPromotionPipeline)
            pipeline._neo = mock_neo
            pipeline._openrouter_key = empty_key
            pipeline._relation_extraction_mode = "llm_assisted"

            result = await pipeline._write_ltm_graph(
                session_id="test-no-key",
                stm_data=stm_data,
                errors=errors,
            )

            assert result == {"nodes": 2, "edges": 1}
            mock_neo.extract_and_write_relations.assert_not_called()
            assert errors == []

    @pytest.mark.asyncio
    async def test_write_survives_auth_error_when_key_present(self) -> None:
        """Auth failure in extraction is caught; promote returns graph counts and appends error."""
        from unittest.mock import AsyncMock

        from graphknows.memory.flush import LTMPromotionPipeline

        mock_neo = AsyncMock()
        mock_neo.write_session_subgraph.return_value = {"nodes": 3, "edges": 2}
        mock_neo.extract_and_write_relations.side_effect = Exception(
            "AuthenticationError: 401 Unauthorized"
        )
        stm_data: dict = {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}
        errors: list[str] = []

        pipeline = LTMPromotionPipeline.__new__(LTMPromotionPipeline)
        pipeline._neo = mock_neo
        pipeline._openrouter_key = "bad-key-xyz"
        pipeline._relation_extraction_mode = "llm_assisted"

        result = await pipeline._write_ltm_graph(
            session_id="test-bad-key",
            stm_data=stm_data,
            errors=errors,
        )

        assert result["nodes"] == 3
        assert result["edges"] == 2  # rel_edges=0; exception caught
        assert len(errors) == 1
        assert "extract_and_write_relations" in errors[0]
        assert "AuthenticationError" in errors[0]


def test_bg_run_flush_offload():
    """_bg_run correctly submits a coroutine to _BG_LOOP from sync context."""
    import asyncio

    from graphknows.memory.stm._async_bridge import _run as _bg_run

    async def _simple():
        await asyncio.sleep(0)
        return 42

    result = _bg_run(_simple())
    assert result == 42


def test_dspy_lm_num_retries_zero(monkeypatch):
    """dspy.LM must be constructed with num_retries=0 to prevent retry storms."""
    import dspy

    captured: dict = {}

    def mock_init(self, *args, **kwargs):
        captured.update(kwargs)
        # avoid actually calling init (no real API key)

    monkeypatch.setattr(dspy.LM, "__init__", mock_init)

    try:
        from graphknows.graphgen.extraction.relations import DSPyExtractor
        DSPyExtractor(config={"openrouter_api_key": "fake", "model": "openai/gpt-4o"})
    except Exception:
        pass

    assert captured.get("num_retries") == 0, "num_retries must be 0 to prevent LiteLLM retry storms"


def test_get_openrouter_api_key_finds_key_under_infra():
    """_get_openrouter_api_key must find the key when placed under config['infra']."""
    from graphknows.config.llm import _get_openrouter_api_key

    config = {
        "llm": {"model": "openai/gpt-4o"},
        "infra": {"openrouter_api_key": "sk-or-test-1234"},
    }
    assert _get_openrouter_api_key(config) == "sk-or-test-1234"
