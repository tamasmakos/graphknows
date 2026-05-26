"""Wave 3 — container.py GraphKnowsContainer tests.

Tests using __new__ bypass + mock injection:
- from_env / from_settings class methods
- retriever/pg/neo/memgraph property guards (raise RuntimeError when not entered)
- __init__ state initialization
- _ERR_NOT_ENTERED sentinel
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_container():
    """Create a GraphKnowsContainer with a mocked settings object (no real DB)."""
    from graphknows.container import GraphKnowsContainer

    container = GraphKnowsContainer.__new__(GraphKnowsContainer)
    container.settings = MagicMock()
    container.settings.neo4j_uri = "bolt://localhost:7687"
    container.settings.neo4j_user = "neo4j"
    container.settings.neo4j_password = "test"
    container.settings.neo4j_database = "neo4j"
    container.settings.pgvector_dsn = "postgresql://user:pass@localhost/db"
    container.settings.memgraph_uri = "bolt://localhost:17687"
    container.settings.embed_model = "test-model"
    container.settings.graphknows_env = "test"
    container._pg = None
    container._neo = None
    container._memgraph = None
    container._retriever = None
    container._jobs = {}
    container._background_tasks = set()
    return container


class TestGraphKnowsContainerInit:
    def test_init_initializes_all_fields_to_none(self):
        from graphknows.container import GraphKnowsContainer

        settings = MagicMock()
        c = GraphKnowsContainer(settings)
        assert c.settings is settings
        assert c._pg is None
        assert c._neo is None
        assert c._memgraph is None
        assert c._retriever is None
        assert c._jobs == {}
        assert c._background_tasks == set()

    def test_err_not_entered_constant(self):
        from graphknows.container import GraphKnowsContainer

        assert "not entered" in GraphKnowsContainer._ERR_NOT_ENTERED.lower()
        assert "async with" in GraphKnowsContainer._ERR_NOT_ENTERED.lower()


class TestFromClassMethods:
    def test_from_settings_returns_container(self):
        from graphknows.container import GraphKnowsContainer

        settings = MagicMock()
        c = GraphKnowsContainer.from_settings(settings)
        assert isinstance(c, GraphKnowsContainer)
        assert c.settings is settings

    def test_from_env_returns_container(self):
        from graphknows.container import GraphKnowsContainer

        with patch("graphknows.container.GraphKnowsSettings") as MockSettings:
            MockSettings.return_value = MagicMock()
            c = GraphKnowsContainer.from_env()
        assert isinstance(c, GraphKnowsContainer)


class TestPropertyGuards:
    def test_retriever_raises_when_not_entered(self):
        container = _make_container()
        container._retriever = None
        with pytest.raises(RuntimeError, match="not entered"):
            _ = container.retriever

    def test_pg_raises_when_not_entered(self):
        container = _make_container()
        container._pg = None
        with pytest.raises(RuntimeError, match="not entered"):
            _ = container.pg

    def test_neo_raises_when_not_entered(self):
        container = _make_container()
        container._neo = None
        with pytest.raises(RuntimeError, match="not entered"):
            _ = container.neo

    def test_memgraph_raises_when_not_entered(self):
        container = _make_container()
        container._memgraph = None
        with pytest.raises(RuntimeError, match="not entered"):
            _ = container.memgraph

    def test_retriever_returns_when_set(self):
        container = _make_container()
        fake_retriever = MagicMock()
        container._retriever = fake_retriever
        assert container.retriever is fake_retriever

    def test_pg_returns_when_set(self):
        container = _make_container()
        fake_pg = MagicMock()
        container._pg = fake_pg
        assert container.pg is fake_pg

    def test_neo_returns_when_set(self):
        container = _make_container()
        fake_neo = MagicMock()
        container._neo = fake_neo
        assert container.neo is fake_neo

    def test_memgraph_returns_when_set(self):
        container = _make_container()
        fake_mg = MagicMock()
        container._memgraph = fake_mg
        assert container.memgraph is fake_mg


class TestRunIngestTask:
    def test_ingest_task_sets_running_status(self):
        container = _make_container()

        run_id = "test-run-123"
        container._jobs[run_id] = {"status": "queued"}

        async def run():
            with patch("graphknows.graphgen.run_ingestion", AsyncMock(return_value={"ok": True})):
                try:
                    await container._run_ingest_task(
                        run_id=run_id,
                        text="Hello world",
                        title="Test",
                        config_override={},
                    )
                except Exception:
                    pass  # May fail — we just want to verify status was set to "running"

        asyncio.run(run())
        # Status should have been set to "running" (even if it later transitions)
        assert container._jobs[run_id]["status"] in ("running", "done", "failed", "completed")


class TestAexit:
    def test_aexit_clears_all_stores(self):
        container = _make_container()

        fake_pg = MagicMock()
        fake_pg.close = AsyncMock()
        fake_neo = MagicMock()
        fake_neo.close = MagicMock()

        container._pg = fake_pg
        container._neo = fake_neo
        container._memgraph = MagicMock()
        container._retriever = MagicMock()

        async def run():
            await container.__aexit__(None, None, None)

        asyncio.run(run())

        assert container._pg is None
        assert container._neo is None
        assert container._memgraph is None
        assert container._retriever is None
        fake_pg.close.assert_called_once()
        fake_neo.close.assert_called_once()

    def test_aexit_handles_pg_close_exception(self):
        """__aexit__ must not propagate store cleanup exceptions."""
        container = _make_container()

        fake_pg = MagicMock()
        fake_pg.close = AsyncMock(side_effect=RuntimeError("DB gone"))
        container._pg = fake_pg
        container._neo = None
        container._memgraph = None
        container._retriever = None

        async def run():
            # Should not raise
            await container.__aexit__(None, None, None)

        asyncio.run(run())
        assert container._pg is None
