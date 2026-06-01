"""Tests for graphknows.config settings."""
from __future__ import annotations
from pathlib import Path
from graphknows.config import GraphKnowsSettings
from graphknows.config.env_loader import load_env_file


def test_settings_defaults():
    s = GraphKnowsSettings()
    assert isinstance(s.neo4j_uri, str)
    assert isinstance(s.memgraph_uri, str)


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
