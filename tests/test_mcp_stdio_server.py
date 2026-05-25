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


def test_all_twelve_tools_registered():
    """All 12 expected tools must be present in the FastMCP app."""
    from graphknows.mcp.stdio_server import app

    # FastMCP exposes list_tools() synchronously or via the tool registry
    tool_names = {t.name for t in app._tool_manager.list_tools()}
    expected = {
        "memory_doctor",
        "memory_stats",
        "memory_ingest",
        "memory_recall",
        "memory_flush",
        "ltm_query",
        "ltm_entity",
        "ltm_topics",
        "recall_by_document",
        "recall_by_dialog",
        "recall_by_topic",
        "corpus_ingest",
    }
    assert expected == tool_names, f"Missing tools: {expected - tool_names}"


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

    def _driver_factory(uri, auth):  # noqa: ANN001
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.retrieval.recall_stm.recall_short_term", return_value=[hit]),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.retrieval.recall_stm.recall_short_term", return_value=[hit]),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.retrieval.recall_stm.recall_short_term", return_value=[]),
    ):
        result = asyncio.run(memory_recall(query="unknown", session_id="test-001"))

    assert result == {"hits": []}


# ---------------------------------------------------------------------------
# LTM session_filter fix (MEM-001)
# ---------------------------------------------------------------------------


def test_ltm_recall_uses_session_filter_none():
    """LTM recall must pass session_filter=None regardless of session_id."""
    import asyncio

    from graphknows.mcp.stdio_server import memory_recall

    hit = MagicMock()
    hit.score = 0.9
    hit.source_type = "chunk"
    hit.text = "some ltm content"
    hit.provenance = {"session_id": "other-session"}
    hit.entity_names = ["test"]

    with (
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.mcp.stdio_server._get_neo", return_value=MagicMock()),
        patch(
            "graphknows.retrieval.recall_ltm.recall_long_term", return_value=[hit]
        ) as mock_recall_ltm,
    ):
        result = asyncio.run(memory_recall(query="test", session_id="current-session", scope="ltm"))

    mock_recall_ltm.assert_called_once()
    _kwargs = mock_recall_ltm.call_args.kwargs
    assert _kwargs.get("session_filter") is None, (
        f"Expected session_filter=None, got {_kwargs.get('session_filter')}"
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
            patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
            patch(
                "graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"
            ),
            patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
            patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
            patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
            patch("graphknows.memory.ingestor._run"),
            patch("graphknows.mcp.stdio_server._get_neo", return_value=MagicMock()),
            patch("graphknows.retrieval.recall_ltm.recall_long_term", return_value=[hit]),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.retrieval.recall_stm.recall_short_term", return_value=[hit]),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.mcp.stdio_server._get_neo", return_value=MagicMock()),
        patch("graphknows.retrieval.recall_ltm.recall_long_term", return_value=[hit]),
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
        patch("graphknows.mcp.stdio_server._mg_uri", return_value="bolt://localhost:17687"),
        patch("graphknows.mcp.stdio_server._pg_dsn", return_value="postgresql://localhost/test"),
        patch("graphknows.mcp.stdio_server._embed_fn", return_value=lambda x: [0.0] * 384),
        patch("graphknows.retrieval.stores.memgraph.MemgraphStore"),
        patch("graphknows.retrieval.stores.pgvector.PgVectorStore"),
        patch("graphknows.memory.ingestor._run"),
        patch("graphknows.mcp.stdio_server._get_neo", return_value=MagicMock()),
        patch("graphknows.retrieval.recall_stm.recall_short_term", return_value=[stm_hit]),
        patch("graphknows.retrieval.recall_ltm.recall_long_term", return_value=[ltm_hit]),
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

    drv = MagicMock()
    drv.session.return_value.__enter__ = lambda s: s
    drv.session.return_value.__exit__ = MagicMock(return_value=False)
    drv.session.return_value.run.return_value.data.return_value = [
        {"community_id": 1, "summary": "AI topics", "entity_count": 5},
        {"community_id": 2, "summary": "ML topics", "entity_count": 3},
    ]

    with (
        patch(
            "graphknows.mcp.stdio_server._neo4j_creds",
            return_value=("bolt://localhost:7687", "neo4j", "pw"),
        ),
        patch("neo4j.GraphDatabase.driver", return_value=drv),
    ):
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

    mock_props = {"name": "TestEntity", "pagerank": 0.5}

    mock_row = MagicMock()
    mock_row.__getitem__.side_effect = lambda k: {"props": mock_props, "segment_count": 3}[k]

    drv = MagicMock()
    drv.session.return_value.__enter__ = lambda s: s
    drv.session.return_value.__exit__ = MagicMock(return_value=False)
    drv.session.return_value.run.return_value.single.return_value = mock_row

    with (
        patch(
            "graphknows.mcp.stdio_server._neo4j_creds",
            return_value=("bolt://localhost:7687", "neo4j", "pw"),
        ),
        patch("neo4j.GraphDatabase.driver", return_value=drv),
    ):
        result = ltm_entity(name="TestEntity")

    assert "error" not in result
    assert result["name"] == "TestEntity"
    assert result["segment_count"] == 3


# ---------------------------------------------------------------------------
# corpus_ingest
# ---------------------------------------------------------------------------


def test_corpus_ingest_calls_run_ingestion():
    """corpus_ingest calls run_ingestion with documents and returns stats."""
    import asyncio

    from graphknows.mcp.stdio_server import corpus_ingest

    mock_stats = {
        "documents_processed": 1,
        "entities_extracted": 5,
        "relations_extracted": 3,
        "nodes_written": 5,
        "edges_written": 3,
        "errors": [],
    }

    with (
        patch(
            "graphknows.graphgen.ingest.run_ingestion",
            new_callable=AsyncMock,
            return_value=mock_stats,
        ) as mock_run,
    ):
        result = asyncio.run(
            corpus_ingest(
                documents=[{"text": "test", "title": "t", "id": "1"}],
            )
        )

    mock_run.assert_called_once_with(
        documents=[{"text": "test", "title": "t", "id": "1"}],
        config_override=None,
    )
    assert result == mock_stats


def test_corpus_ingest_empty_documents():
    """corpus_ingest returns error when documents list is empty."""
    import asyncio

    from graphknows.mcp.stdio_server import corpus_ingest

    result = asyncio.run(corpus_ingest(documents=[]))

    assert "error" in result
    assert result["documents_processed"] == 0
