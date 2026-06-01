"""Unit tests for the graphknows package public surface.

Covers settings without touching any external services
(no DB, no network, no filesystem state).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Settings (no side effects — import only, no live env required)
# ---------------------------------------------------------------------------


class TestGraphKnowsSettings:
    """GraphKnowsSettings can be instantiated with env-var overrides."""

    def test_settings_accept_explicit_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PGVECTOR_DSN", "postgresql://u:p@localhost:5432/testdb")
        monkeypatch.setenv("PGVECTOR_PASSWORD", "pw")
        monkeypatch.setenv("NEO4J_PASSWORD", "neo")
        monkeypatch.setenv("MEMGRAPH_PASSWORD", "mg")
        monkeypatch.setenv("GRAPHKNOWS_ENV", "test")

        from graphknows.config import GraphKnowsSettings

        settings = GraphKnowsSettings()
        assert settings.pgvector_dsn == "postgresql://u:p@localhost:5432/testdb"

    def test_settings_graphknows_env_field_is_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GRAPHKNOWS_ENV", "test")
        monkeypatch.setenv("PGVECTOR_DSN", "postgresql://u:p@localhost:5432/testdb")
        monkeypatch.setenv("PGVECTOR_PASSWORD", "pw")
        monkeypatch.setenv("NEO4J_PASSWORD", "neo")
        monkeypatch.setenv("MEMGRAPH_PASSWORD", "mg")

        from graphknows.config import GraphKnowsSettings

        settings = GraphKnowsSettings()
        assert settings.graphknows_env == "test"


# ===========================================================================
# Config settings: infrastructure, LLM, ontology, extraction, processing
# ===========================================================================


import os
import pytest

