"""Unit tests for the graphknows package public surface.

Covers constants, entity labels, and settings without touching any
external services (no DB, no network, no filesystem state).
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestEmbeddingConstants:
    """Constants declared in graphknows.models.constants."""

    def test_embedding_model_is_bge_small(self):
        from graphknows.models.constants import EMBEDDING_MODEL

        assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

    def test_embedding_dim_is_384(self):
        from graphknows.models.constants import EMBEDDING_DIM

        assert EMBEDDING_DIM == 384


# ---------------------------------------------------------------------------
# EntityLabel enum
# ---------------------------------------------------------------------------


class TestEntityLabel:
    """EntityLabel(StrEnum) values and membership."""

    def test_person_value(self):
        from graphknows.models.constants import EntityLabel

        assert EntityLabel.PERSON == "Person"
        assert str(EntityLabel.PERSON) == "Person"

    def test_all_labels_are_strings(self):
        from graphknows.models.constants import EntityLabel

        for label in EntityLabel:
            assert isinstance(label, str), f"{label!r} should be a str"

    def test_expected_labels_present(self):
        from graphknows.models.constants import EntityLabel

        expected = {
            "Person",
            "Organization",
            "Policy",
            "Treaty",
            "Legislation",
            "Concept",
            "Event",
            "Place",
            "Regulation",
            "Topic",
        }
        assert {str(e) for e in EntityLabel} == expected

    def test_membership_check(self):
        from graphknows.models.constants import EntityLabel

        assert "Person" in EntityLabel
        assert "NotALabel" not in EntityLabel


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
