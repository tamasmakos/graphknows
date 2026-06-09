"""Tests for the graphknows MCP stdio server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
        "memory_ingest_turn",
        "memory_recall",
        "memory_flush",
        "memory_mirror",
        "ltm_query",
        "ltm_entity",
        "ltm_topics",
        "recall_by_document",
        "recall_by_dialog",
        "recall_by_topic",
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch(
            "graphknows.mcp.stdio_server._neo4j_creds",
            return_value=("bolt://localhost:7687", "neo4j", "pw"),
        ),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch(
            "graphknows.mcp.stdio_server._neo4j_creds",
            return_value=("bolt://localhost:7687", "neo4j", "pw"),
        ),
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

    hit = MagicMock()
    hit.score = 0.95
    hit.source_type = "chunk"
    hit.text = "graphknows is a framework-agnostic memory layer."
    hit.provenance = {"session_id": "test-001"}
    hit.entity_names = ["graphknows"]

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[hit]),
    ):
        result = asyncio.run(memory_recall(query="graphknows", session_id="test-001", top_k=5))

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

    hit = MagicMock()
    hit.score = 0.95
    hit.cosine_score = 0.87
    hit.source_type = "chunk"
    hit.text = "graphknows is a framework-agnostic memory layer."
    hit.provenance = {"session_id": "test-001"}
    hit.entity_names = ["graphknows"]

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[hit]),
    ):
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

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[]),
    ):
        result = asyncio.run(memory_recall(query="unknown", session_id="test-001"))

    assert result == {"hits": []}


# ---------------------------------------------------------------------------
# LTM session_filter fix (MEM-001)
# ---------------------------------------------------------------------------


def test_ltm_recall_uses_session_filter():
    """LTM recall must scope results to the requested session_id."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    hit = MagicMock()
    hit.score = 0.9
    hit.source_type = "chunk"
    hit.text = "some ltm content"
    hit.provenance = {"session_id": "other-session"}
    hit.entity_names = ["test"]

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.get_pg_store_bridge", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.get_neo_store", return_value=MagicMock()),
        patch(
            "graphknows.memory.query.recall_ltm.recall_long_term", return_value=[hit]
        ) as mock_recall_ltm,
    ):
        result = asyncio.run(memory_recall(query="test", session_id="current-session", scope="ltm"))

    mock_recall_ltm.assert_called_once()
    _kwargs = mock_recall_ltm.call_args.kwargs
    assert _kwargs.get("session_filter") == "current-session", (
        f"Expected session_filter='current-session', got {_kwargs.get('session_filter')}"
    )
    assert len(result["hits"]) == 1


# ---------------------------------------------------------------------------
# Content prefix strip fix (MEM-002)
# ---------------------------------------------------------------------------


def test_clean_strips_content_prefix():
    """_clean() strips the '## Content\\n' heading artifact (case-insensitive)."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    def _run_ltm(text: str, score: float = 0.9) -> str:
        hit = MagicMock()
        hit.score = score
        hit.source_type = "chunk"
        hit.text = text
        hit.provenance = {"session_id": "s1"}
        hit.entity_names = []

        with (
            patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
            patch("graphknows.mcp.tools.stm.get_pg_store", new=AsyncMock(return_value=MagicMock())),
            patch("graphknows.mcp.tools.stm.get_neo_store", return_value=MagicMock()),
            patch("graphknows.memory.query.recall_ltm.recall_long_term", return_value=[hit]),
        ):
            return asyncio.run(memory_recall(query="x", session_id="s1", scope="ltm"))["hits"][0][
                "text"
            ]

    assert _run_ltm("## Content\n\nSome real content here.") == "Some real content here."
    assert _run_ltm("## content\n\nLower case.") == "Lower case."
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
    """scope='stm' must behave identically (no normalise/weight applied)."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    hit = MagicMock()
    hit.score = 0.95
    hit.source_type = "chunk"
    hit.text = "stm only content"
    hit.provenance = {"session_id": "test-001"}
    hit.entity_names = []

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[hit]),
    ):
        result = asyncio.run(memory_recall(query="x", session_id="test-001", scope="stm"))

    assert len(result["hits"]) == 1
    assert result["hits"][0]["score"] == round(0.95, 6)


def test_scope_ltm_unchanged():
    """scope='ltm' must behave identically (no normalise/weight applied)."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    hit = MagicMock()
    hit.score = 0.87
    hit.source_type = "chunk"
    hit.text = "ltm only content"
    hit.provenance = {"session_id": "other"}
    hit.entity_names = []

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.get_pg_store_bridge", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.get_neo_store", return_value=MagicMock()),
        patch("graphknows.memory.query.recall_ltm.recall_long_term", return_value=[hit]),
    ):
        result = asyncio.run(memory_recall(query="x", session_id="test-001", scope="ltm"))

    assert len(result["hits"]) == 1
    assert result["hits"][0]["score"] == round(0.87, 6)


def test_scope_both_normalises_cross_scope_scores():
    """scope='both' normalises STM/LTM scores before merge and respects weights."""
    import asyncio
    from types import SimpleNamespace

    from graphknows.mcp.stdio_server import memory_recall

    stm_hit = SimpleNamespace()
    stm_hit.score = 0.016
    stm_hit.cosine_score = 0.016
    stm_hit.source_type = "chunk"
    stm_hit.text = "stm content"
    stm_hit.provenance = {"session_id": "s1"}
    stm_hit.entity_names = []

    ltm_hit = SimpleNamespace()
    ltm_hit.score = 0.033
    ltm_hit.cosine_score = 0.033
    ltm_hit.source_type = "chunk"
    ltm_hit.text = "ltm content"
    ltm_hit.provenance = {"session_id": "s2"}
    ltm_hit.entity_names = []

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.mcp.tools.stm.get_pg_store_bridge", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.get_neo_store", return_value=MagicMock()),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[stm_hit]),
        patch("graphknows.memory.query.recall_ltm.recall_long_term", return_value=[ltm_hit]),
    ):
        result = asyncio.run(memory_recall(query="x", session_id="s1", top_k=5, scope="both"))

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
        "recall_by_document",
        "recall_by_dialog",
        "recall_by_topic",
    ):
        assert name in tool_names, f"Missing tool: {name}"


def test_ltm_topics_returns_topics():
    """ltm_topics returns topic list above threshold."""
    from graphknows.mcp.stdio_server import ltm_topics

    mock_store = MagicMock()
    mock_store.list_topic_nodes.return_value = [
        {"community_id": 1, "summary": "AI topics", "entity_count": 5},
        {"community_id": 2, "summary": "ML topics", "entity_count": 3},
    ]

    with patch("graphknows.mcp.tools.ltm.get_neo_store", return_value=mock_store):
        result = ltm_topics(min_entities=3)

    assert "topics" in result
    assert len(result["topics"]) == 2
    assert result["topics"][0]["community_id"] == 1
    assert result["topics"][0]["entity_count"] == 5


def test_ltm_entity_not_found():
    """ltm_entity returns error dict when entity not found."""
    from graphknows.mcp.stdio_server import ltm_entity

    drv = MagicMock()
    drv.session.return_value.__enter__ = lambda s: s
    drv.session.return_value.__exit__ = MagicMock(return_value=False)
    drv.session.return_value.run.return_value.single.return_value = None

    with (
        patch(
            "graphknows.mcp.stdio_server._neo4j_creds",
            return_value=("bolt://localhost:7687", "neo4j", "pw"),
        ),
        patch("neo4j.GraphDatabase.driver", return_value=drv),
    ):
        result = ltm_entity(name="NonExistent")

    assert "error" in result
    assert "NonExistent" in result["error"]


def test_ltm_entity_found():
    """ltm_entity returns entity properties when found."""
    from graphknows.mcp.stdio_server import ltm_entity

    mock_props = {"name": "TestEntity", "pagerank": 0.5, "segment_count": 3}

    mock_store = MagicMock()
    mock_store.get_entity_by_name.return_value = mock_props

    with patch("graphknows.mcp.tools.ltm.get_neo_store", return_value=mock_store):
        result = ltm_entity(name="TestEntity")

    assert "error" not in result
    assert result["name"] == "TestEntity"
    assert result["segment_count"] == 3


# ---------------------------------------------------------------------------
# corpus_ingest
# ---------------------------------------------------------------------------


def test_corpus_ingest_calls_run_ingestion():
    """corpus_ingest routes through CorpusIngestWorkflow (not graphgen.run_ingestion)."""
    import asyncio

    from graphknows.mcp.tools.corpus import corpus_ingest

    fake_result = MagicMock()
    fake_result.session_id = "sess-abc"
    fake_result.documents_processed = 1
    fake_result.entities_extracted = 5
    fake_result.chunks_written = 2
    fake_result.nodes_written = 5
    fake_result.edges_written = 3
    fake_result.elapsed_s = 0.1
    fake_result.errors = []

    mock_workflow = MagicMock()
    mock_workflow.run = AsyncMock(return_value=fake_result)

    with (
        patch(
            "graphknows.mcp.tools.corpus.get_settings",
            return_value=MagicMock(),
        ),
        patch(
            "graphknows.config.builders.stores.build_corpus_workflow",
            return_value=mock_workflow,
        ),
    ):
        result = asyncio.run(
            corpus_ingest(
                documents=[{"text": "test", "title": "t", "id": "1"}],
                session_id="sess-abc",
            )
        )

    assert result["documents_processed"] == 1
    assert result["nodes_written"] == 5
    assert "session_id" in result
    request = mock_workflow.run.await_args.args[0]
    assert request.session_id == "sess-abc"
    assert request.summarize_topics is True


def test_corpus_ingest_empty_documents():
    """corpus_ingest returns error when documents list is empty."""
    import asyncio

    from graphknows.mcp.stdio_server import corpus_ingest

    result = asyncio.run(corpus_ingest(documents=[]))

    assert "error" in result
    assert result["documents_processed"] == 0


# ===========================================================================
# MCP helpers: mg_uri, pg_dsn, neo4j_creds
# ===========================================================================


import asyncio
import os

import pytest

# ---------------------------------------------------------------------------
# _mg_uri / _pg_dsn / _neo4j_creds — env var readers
# ---------------------------------------------------------------------------


class TestMgUri:
    def test_returns_default_when_env_not_set(self):
        import graphknows.mcp.stdio_server as srv

        env = {k: v for k, v in os.environ.items() if k != "MEMGRAPH_URI"}
        with patch.dict(os.environ, env, clear=True):
            result = srv._mg_uri()
        assert "bolt://" in result

    def test_returns_env_value_when_set(self):
        import graphknows.mcp.stdio_server as srv

        with patch.dict(os.environ, {"MEMGRAPH_URI": "bolt://myhost:1234"}):
            result = srv._mg_uri()
        assert result == "bolt://myhost:1234"


class TestPgDsn:
    def test_returns_default_postgresql_dsn(self):
        import graphknows.mcp.stdio_server as srv

        env = {k: v for k, v in os.environ.items() if k != "PGVECTOR_DSN"}
        with patch.dict(os.environ, env, clear=True):
            result = srv._pg_dsn()
        assert result.startswith("postgresql://")

    def test_returns_env_value_when_set(self):
        import graphknows.mcp.stdio_server as srv

        custom = "postgresql://user:pass@db:5432/mydb"
        with patch.dict(os.environ, {"PGVECTOR_DSN": custom}):
            result = srv._pg_dsn()
        assert result == custom


class TestNeo4jCreds:
    def test_returns_tuple_of_three(self):
        import graphknows.mcp.stdio_server as srv

        uri, user, pw = srv._neo4j_creds()
        assert isinstance(uri, str)
        assert isinstance(user, str)
        assert isinstance(pw, str)

    def test_returns_env_values_when_set(self):
        import graphknows.mcp.stdio_server as srv

        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://neo4j-host:7687",
                "NEO4J_USER": "admin",
                "NEO4J_PASSWORD": "secret",
            },
        ):
            uri, user, pw = srv._neo4j_creds()
        assert uri == "bolt://neo4j-host:7687"
        assert user == "admin"
        assert pw == "secret"

    def test_returns_defaults_without_env(self):
        import graphknows.mcp.stdio_server as srv

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        }
        with patch.dict(os.environ, env, clear=True):
            uri, user, pw = srv._neo4j_creds()
        assert "bolt://" in uri
        assert user == "neo4j"


# ---------------------------------------------------------------------------
# _get_mg — lazy singleton
# ---------------------------------------------------------------------------


class TestGetMg:
    def test_caches_instance_on_second_call(self):
        import graphknows.mcp.stdio_server as srv

        # Pre-set singleton
        fake_instance = MagicMock()
        srv._mg = fake_instance
        result = srv._get_mg()
        assert result is fake_instance
        # Restore
        srv._mg = None


# ---------------------------------------------------------------------------
# _get_pg — async lazy singleton
# ---------------------------------------------------------------------------


class TestGetPg:
    def test_caches_instance_when_already_connected(self):
        import graphknows.mcp.stdio_server as srv

        fake_pg = MagicMock()
        srv._pg = fake_pg
        srv._pg_connected = True

        async def run():
            result = await srv._get_pg()
            return result

        result = asyncio.run(run())
        assert result is fake_pg
        # Restore
        srv._pg = None
        srv._pg_connected = False


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

        # Mock get_neo_store() to return a Neo4j store whose _driver.session.run returns count > 0
        mock_neo = MagicMock()
        mock_neo._db = "neo4j"
        mock_neo_session = MagicMock()
        mock_neo._driver.session.return_value.__enter__.return_value = mock_neo_session
        # MATCH (s:SESSION {id: $sid}) RETURN count(s) AS n  →  n=1
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, k: 1
        mock_neo_session.run.return_value = [mock_record]

        mock_flusher_cls = MagicMock()

        with (
            patch.object(stm_mod, "get_neo_store", return_value=mock_neo),
            patch.dict(
                "sys.modules",
                {
                    "graphknows.memory.flush": MagicMock(
                        FlushOptions=MagicMock(), SessionFlusher=mock_flusher_cls
                    ),
                },
            ),
        ):
            result = await stm_mod.memory_flush("session-already-flushed")

        assert result["status"] == "already_flushed"
        assert result["session_id"] == "session-already-flushed"
        assert result["nodes_written"] == 0
        assert result["errors"] == []
        # Flusher should NOT have been instantiated
        mock_flusher_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_session_calls_flusher(self):
        """When session is NOT in LTM, the flusher should be called."""
        from graphknows.mcp.tools import stm as stm_mod

        # Neo4j session returns count = 0 (not in LTM)
        mock_neo = MagicMock()
        mock_neo._db = "neo4j"
        mock_neo_session = MagicMock()
        mock_neo._driver.session.return_value.__enter__.return_value = mock_neo_session
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, k: 0
        mock_neo_session.run.return_value = [mock_record]

        # Mock everything needed for flush
        mock_result = MagicMock()
        mock_result.nodes_written = 5
        mock_result.edges_written = 3
        mock_result.chunks_promoted = 2
        mock_result.turns_promoted = 1
        mock_result.elapsed_s = 0.5
        mock_result.errors = []

        mock_pipeline_instance = AsyncMock()
        mock_pipeline_instance.promote = AsyncMock(return_value=mock_result)

        with (
            patch.object(stm_mod, "get_neo_store", return_value=mock_neo),
            patch("graphknows.config.builders.stores.build_promotion_pipeline", return_value=mock_pipeline_instance),
        ):
            result = await stm_mod.memory_flush("session-new")

        # Pipeline MUST have been called
        mock_pipeline_instance.promote.assert_awaited_once()
        # Result should NOT have already_flushed status
        assert result.get("status") != "already_flushed"


def test_memory_ingest_returns_no_warnings():
    """memory_ingest must not return rel_extractor_disabled warning — issue #113."""
    import asyncio
    from graphknows.mcp.stdio_server import memory_ingest

    mock_result = MagicMock()
    mock_result.file_id = "abc123"
    mock_result.chunk_count = 1
    mock_result.entity_count = 5
    mock_result.elapsed_s = 1.0
    mock_result.errors = []

    with (
        patch("graphknows.config.builders.stores.build_stm_service") as mock_build,
        patch("graphknows.mcp.tools.stm.get_settings"),
    ):
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = mock_result
        mock_build.return_value = mock_ingestor
        result = asyncio.run(memory_ingest("some text", session_id="test", title="t"))
        assert result["warnings"] == [], f"Expected no warnings, got: {result['warnings']}"
        assert "rel_extractor_disabled" not in result["warnings"]


def test_memory_flush_summarizes_topics():
    """memory_flush must always request topic summarization (run_topics=True by default)."""
    import asyncio
    from graphknows.mcp.stdio_server import memory_flush

    captured_request = {}

    async def fake_promote(request):
        captured_request["run_topics"] = request.run_topics
        m = MagicMock()
        m.nodes_written = 1
        m.edges_written = 1
        m.chunks_promoted = 1
        m.turns_promoted = 0
        m.elapsed_s = 1.0
        m.errors = []
        return m

    mock_neo = MagicMock()
    mock_neo_sess = MagicMock()
    mock_neo_sess.run.return_value = [{"n": 0}]
    mock_neo._driver.session.return_value.__enter__ = lambda s: mock_neo_sess
    mock_neo._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("graphknows.config.builders.stores.build_promotion_pipeline") as mock_build,
        patch("graphknows.mcp.tools.stm.get_settings"),
        patch("graphknows.mcp.tools.stm.get_neo_store", return_value=mock_neo),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.promote = fake_promote
        mock_build.return_value = mock_pipeline
        asyncio.run(memory_flush("test-session"))
        assert captured_request.get("run_topics") is True, \
            "memory_flush must use run_topics=True (PromotionRequest default)"


# ---------------------------------------------------------------------------
# Issue #111 — STM recall uses _BG_LOOP pg store, not get_pg_store()
# ---------------------------------------------------------------------------


def test_stm_recall_uses_bg_loop_pg_store():
    """memory_recall builds a fresh pg store on _BG_LOOP for stm scope, not get_pg_store()."""
    import asyncio

    from graphknows.mcp.tools.stm import memory_recall

    with (
        patch("graphknows.mcp.tools.stm.get_pg_store") as mock_get_pg,
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store") as mock_build,
        patch("graphknows.mcp.tools.stm._bg_run") as mock_bg_run,
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[]),
    ):
        asyncio.run(memory_recall("test query", session_id="test-session", scope="stm"))
        mock_get_pg.assert_not_called()
        mock_build.assert_called_once()
        mock_bg_run.assert_called_once()


def test_stm_recall_hit_entity_only_has_cosine_score_zero():
    """Entity-only recall hits must have cosine_score explicitly set to 0.0 — issue #112."""
    import asyncio
    from graphknows.mcp.stdio_server import memory_recall

    entity_hit = MagicMock()
    entity_hit.score = 0.016
    entity_hit.cosine_score = 0.0
    entity_hit.source_type = "chunk"
    entity_hit.text = "entity-only hit"
    entity_hit.provenance = {"session_id": "test-001"}
    entity_hit.entity_names = ["graphknows"]

    with (
        patch("graphknows.mcp.tools.stm.get_mg_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.mcp.tools.stm._bg_run"),
        patch("graphknows.memory.stm.recall.recall_short_term", return_value=[entity_hit]),
    ):
        result = asyncio.run(memory_recall(query="test", session_id="test-001", scope="stm"))
    hits = result.get("hits", [])
    assert len(hits) == 1
    assert hits[0]["cosine_score"] == 0.0


class TestLTMWriterRelationExtractionGuard:
    """LTMPromotionPipeline._write_ltm_graph must skip/survive relation extraction failures."""

    @pytest.mark.asyncio
    async def test_write_skips_extraction_when_key_empty(self) -> None:
        """Empty or whitespace-only openrouter_key → extract_and_write_relations is never called."""
        from unittest.mock import AsyncMock, MagicMock

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
