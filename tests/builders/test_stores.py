"""Smoke tests for graphknows.builders store factories."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphknows.builders import build_memgraph_store, build_neo4j_store, build_pgvector_store
from graphknows.config import GraphKnowsSettings


def test_build_neo4j_store_smoke():
    settings = GraphKnowsSettings()
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        store = build_neo4j_store(settings)
        assert store is not None


def test_build_pgvector_store_smoke():
    settings = GraphKnowsSettings()
    store = build_pgvector_store(settings)
    assert store is not None
    assert store.dsn == settings.pgvector_dsn


def test_build_memgraph_store_smoke():
    settings = GraphKnowsSettings()
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        store = build_memgraph_store(settings)
        assert store is not None
