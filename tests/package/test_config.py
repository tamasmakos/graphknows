"""Unit tests for the graphknows package public surface.

Covers constants, entity labels, and settings without touching any
external services (no DB, no network, no filesystem state).
"""

from __future__ import annotations

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

        # Use value iteration (compatible with Python 3.11 — EnumType.__contains__
        # accepting raw strings requires Python 3.12+)
        values = {e.value for e in EntityLabel}
        assert "Person" in values
        assert "NotALabel" not in values


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


# ---------------------------------------------------------------------------
# InfrastructureSettings
# ---------------------------------------------------------------------------

class TestInfrastructureSettings:
    @pytest.fixture(autouse=True)
    def set_required_env(self, monkeypatch):
        """Ensure PGVECTOR_PASSWORD is always set for infra tests."""
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpassword")

    def test_defaults(self, monkeypatch):
        from graphknows.graphgen.config.settings import InfrastructureSettings
        cfg = InfrastructureSettings()
        # Verify the fields exist and have expected types (values may be overridden by real .env)
        assert isinstance(cfg.neo4j_host, str)
        assert isinstance(cfg.neo4j_port, int)
        assert isinstance(cfg.neo4j_user, str)
        assert isinstance(cfg.neo4j_database, str)

    def test_neo4j_uri_property(self, monkeypatch):
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpassword")
        from graphknows.graphgen.config.settings import InfrastructureSettings
        cfg = InfrastructureSettings(NEO4J_HOST="myhost", NEO4J_PORT="9999")
        assert cfg.neo4j_uri == "bolt://myhost:9999"

    def test_pgvector_password_validator_rejects_empty(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_PASSWORD", raising=False)
        from graphknows.graphgen.config.settings import InfrastructureSettings
        with pytest.raises(Exception):
            InfrastructureSettings(PGVECTOR_PASSWORD="")

    def test_pgvector_password_validator_rejects_changeme(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_PASSWORD", raising=False)
        from graphknows.graphgen.config.settings import InfrastructureSettings
        with pytest.raises(Exception):
            InfrastructureSettings(PGVECTOR_PASSWORD="changeme")

    def test_pgvector_password_validator_accepts_real(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_PASSWORD", raising=False)
        from graphknows.graphgen.config.settings import InfrastructureSettings
        cfg = InfrastructureSettings(PGVECTOR_PASSWORD="s3cr3t!")
        assert cfg.pgvector_password == "s3cr3t!"

    def test_pgvector_dsn_masked(self, monkeypatch):
        """
        NOTE: This test documents a KNOWN BUG in InfrastructureSettings.pgvector_dsn.
        The property docstring claims to mask the password with ***, but the
        implementation embeds the real password in the URL. This test confirms
        the current (buggy) behavior so a future fix is visible as a test change.
        """
        monkeypatch.setenv("PGVECTOR_PASSWORD", "mypassword")
        from graphknows.graphgen.config.settings import InfrastructureSettings
        cfg = InfrastructureSettings()
        dsn = cfg.pgvector_dsn
        # BUG: real password is exposed in the DSN string (should be masked)
        assert "mypassword" in dsn  # Documents current buggy behavior


# ---------------------------------------------------------------------------
# LLMSettings
# ---------------------------------------------------------------------------

class TestLLMSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import LLMSettings
        cfg = LLMSettings()
        assert cfg.temperature == 0.0
        assert cfg.max_retries == 3

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        from graphknows.graphgen.config.settings import LLMSettings
        cfg = LLMSettings()
        assert cfg.base_model == "gpt-4"


# ---------------------------------------------------------------------------
# OntologySettings
# ---------------------------------------------------------------------------

class TestOntologySettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import OntologySettings
        cfg = OntologySettings()
        assert cfg.enabled is False
        assert cfg.merge_with_manual is True
        assert cfg.top_level_only is True


# ---------------------------------------------------------------------------
# ExtractionSettings
# ---------------------------------------------------------------------------

class TestExtractionSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import ExtractionSettings
        cfg = ExtractionSettings()
        assert cfg.chunk_size == 1200
        assert cfg.chunk_overlap == 100
        assert cfg.ner_backend == "gliner2"
        assert cfg.entity_labels == []

    def test_backend_property_mirrors_ner_backend(self):
        from graphknows.graphgen.config.settings import ExtractionSettings
        cfg = ExtractionSettings(ner_backend="spacy")
        assert cfg.backend == "spacy"

    def test_entity_labels_validator_accepts_list(self):
        from graphknows.graphgen.config.settings import ExtractionSettings
        cfg = ExtractionSettings(gliner_labels=["PERSON", "ORG"])
        assert cfg.entity_labels == ["PERSON", "ORG"]

    def test_entity_labels_validator_accepts_none(self):
        from graphknows.graphgen.config.settings import ExtractionSettings
        cfg = ExtractionSettings(gliner_labels=None)
        assert cfg.entity_labels == []

    def test_normalize_backend_fields_compat(self):
        from graphknows.graphgen.config.settings import ExtractionSettings
        cfg = ExtractionSettings.model_validate({"backend": "custom_ner"})
        assert cfg.ner_backend == "custom_ner"


# ---------------------------------------------------------------------------
# ProcessingSettings
# ---------------------------------------------------------------------------

class TestProcessingSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import ProcessingSettings
        cfg = ProcessingSettings()
        assert cfg.enable_pruning is True
        assert cfg.similarity_threshold == 0.95
        assert cfg.min_component_size == 3


# ---------------------------------------------------------------------------
# EmbeddingSettings
# ---------------------------------------------------------------------------

class TestEmbeddingSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import EmbeddingSettings
        cfg = EmbeddingSettings()
        assert "bge" in cfg.model_name.lower()
        assert cfg.batch_size == 32


# ---------------------------------------------------------------------------
# CommunitySettings
# ---------------------------------------------------------------------------

class TestCommunitySettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import CommunitySettings
        cfg = CommunitySettings()
        assert len(cfg.resolutions) > 0
        assert cfg.min_community_size == 3
        assert cfg.node2vec_enabled is False

    def test_resolutions_are_floats(self):
        from graphknows.graphgen.config.settings import CommunitySettings
        cfg = CommunitySettings()
        for r in cfg.resolutions:
            assert isinstance(r, float)


# ---------------------------------------------------------------------------
# AnalyticsSettings
# ---------------------------------------------------------------------------

class TestAnalyticsSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import AnalyticsSettings
        cfg = AnalyticsSettings()
        assert cfg.enabled is False
        assert cfg.silhouette_min_samples == 3


# ---------------------------------------------------------------------------
# TestModeSettings
# ---------------------------------------------------------------------------

class TestTestModeSettings:
    def test_defaults(self):
        from graphknows.graphgen.config.settings import TestModeSettings
        cfg = TestModeSettings()
        assert cfg.enabled is False
        assert cfg.max_documents == 0
        assert cfg.max_chunks == 0


# ---------------------------------------------------------------------------
# PipelineSettings (master object)
# ---------------------------------------------------------------------------

class TestPipelineSettings:
    def test_default_construction(self, monkeypatch):
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpw")
        from graphknows.graphgen.config.settings import PipelineSettings
        # The infra sub-settings will pick up the env var
        cfg = PipelineSettings()
        assert cfg.debug is False
        assert cfg.extraction.chunk_size == 1200

    def test_analysis_analytics_alias_sync(self, monkeypatch):
        """analytics field synced to analysis when analytics provided."""
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpw")
        from graphknows.graphgen.config.settings import PipelineSettings, AnalyticsSettings
        analytics_data = AnalyticsSettings(ANALYTICS_ENABLED=True)
        cfg = PipelineSettings.model_validate({"analytics": analytics_data.model_dump()})
        # Both aliases should resolve to the same enabled state
        assert cfg.analytics.enabled is True
        assert cfg.analysis.enabled is True

    def test_schema_config_has_nodes(self, monkeypatch):
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpw")
        from graphknows.graphgen.config.settings import PipelineSettings
        cfg = PipelineSettings()
        assert "nodes" in cfg.schema_config
        assert len(cfg.schema_config["nodes"]) > 0
