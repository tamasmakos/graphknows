"""Tests for graphknows.config settings."""
from __future__ import annotations
from pathlib import Path

from pydantic import ValidationError

from graphknows.config import GraphKnowsSettings
from graphknows.config.env_loader import load_env_file


def test_settings_defaults():
    s = GraphKnowsSettings()
    assert isinstance(s.neo4j_uri, str)
    assert isinstance(s.memgraph_uri, str)
    assert s.relation_extraction_mode == "gliner2"


def test_load_env_file_returns_empty_for_missing_path():
    result = load_env_file(None)
    assert result == {}


def test_load_env_file_reads_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY=hello\nANOTHER=world\n")
    result = load_env_file(env_file)
    assert result.get("TEST_KEY") == "hello"
    assert result.get("ANOTHER") == "world"


def test_settings_from_env_var(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://test-host:7687")
    s = GraphKnowsSettings()
    assert s.neo4j_uri == "bolt://test-host:7687"


def test_relation_extraction_mode_from_env(monkeypatch):
    monkeypatch.setenv("RELATION_EXTRACTION_MODE", "llm_assisted")
    s = GraphKnowsSettings()
    assert s.relation_extraction_mode == "llm_assisted"


def test_invalid_relation_extraction_mode_rejected(monkeypatch):
    monkeypatch.setenv("RELATION_EXTRACTION_MODE", "ambient_llm")
    try:
        GraphKnowsSettings()
    except ValidationError as exc:
        assert "RELATION_EXTRACTION_MODE" in str(exc) or "relation_extraction_mode" in str(exc)
    else:
        raise AssertionError("invalid relation extraction mode was accepted")
