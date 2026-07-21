"""Tests for pure utility functions in graphknows/graphgen/."""

from __future__ import annotations

import os
import tempfile

import networkx as nx

from graphknows.ingestion.extraction.entities.graph_ops import (
    create_output_directory,
    merge_node_into,
    standardize_label,
)


# ---------------------------------------------------------------------------
# standardize_label
# ---------------------------------------------------------------------------


def test_standardize_label_two_words_with_space():
    assert standardize_label("Political Party") == "POLITICAL_PARTY"


def test_standardize_label_with_mixed_case_and_space():
    assert standardize_label("EU Commission") == "EU_COMMISSION"


def test_standardize_label_empty_string_returns_empty():
    assert standardize_label("") == ""


def test_standardize_label_hyphen_replaced_by_underscore():
    assert standardize_label("Hello-World") == "HELLO_WORLD"


def test_standardize_label_single_word_uppercased():
    assert standardize_label("abc") == "ABC"


def test_standardize_label_strips_leading_trailing_underscores():
    # Leading/trailing non-word chars become underscores, then stripped
    assert standardize_label("  spaces  ") == "SPACES"


def test_standardize_label_already_uppercase_unchanged():
    assert standardize_label("HELLO") == "HELLO"


def test_standardize_label_numbers_preserved():
    assert standardize_label("Article 50") == "ARTICLE_50"


# ---------------------------------------------------------------------------
# merge_node_into
# ---------------------------------------------------------------------------


def _simple_graph():
    g = nx.DiGraph()
    g.add_node("A", name="Alpha")
    g.add_node("B", name="Beta")
    g.add_node("C", name="Gamma")
    g.add_edge("A", "B", weight=1.0)
    return g


def test_merge_node_into_removes_source_node():
    g = _simple_graph()
    merge_node_into(g, "A", "B")
    assert "A" not in g.nodes


def test_merge_node_into_target_node_still_exists():
    g = _simple_graph()
    merge_node_into(g, "A", "B")
    assert "B" in g.nodes


def test_merge_node_into_transfers_incoming_edges_to_target():
    g = nx.DiGraph()
    g.add_node("src")
    g.add_node("tgt")
    g.add_node("pred")
    g.add_edge("pred", "src", weight=0.5)
    merge_node_into(g, "src", "tgt")
    assert g.has_edge("pred", "tgt")
    assert not g.has_edge("pred", "src")


def test_merge_node_into_transfers_outgoing_edges_to_target():
    g = nx.DiGraph()
    g.add_node("src")
    g.add_node("tgt")
    g.add_node("succ")
    g.add_edge("src", "succ", weight=0.7)
    merge_node_into(g, "src", "tgt")
    assert g.has_edge("tgt", "succ")
    assert not g.has_edge("src", "succ")


def test_merge_node_into_source_not_in_graph_no_crash():
    g = _simple_graph()
    # Should not raise; graph unchanged
    merge_node_into(g, "NONEXISTENT", "B")
    assert set(g.nodes) == {"A", "B", "C"}


def test_merge_node_into_target_not_in_graph_no_crash():
    g = _simple_graph()
    merge_node_into(g, "A", "NONEXISTENT")
    assert "A" in g.nodes


def test_merge_node_into_self_loop_on_source_transferred_to_target():
    g = nx.DiGraph()
    g.add_node("src")
    g.add_node("tgt")
    g.add_edge("src", "src", weight=1.0)  # self-loop on source
    merge_node_into(g, "src", "tgt")
    assert not g.has_node("src")
    assert g.has_edge("tgt", "tgt")


def test_merge_node_into_source_alias_added_to_target():
    g = nx.DiGraph()
    g.add_node("Alpha", name="Alpha")
    g.add_node("Beta", name="Beta", aliases=[])
    merge_node_into(g, "Alpha", "Beta")
    aliases = g.nodes["Beta"].get("aliases", [])
    assert "Alpha" in aliases


def test_merge_node_into_weight_takes_max_when_edge_exists():
    g = nx.DiGraph()
    g.add_node("src")
    g.add_node("tgt")
    g.add_node("pred")
    g.add_edge("pred", "src", weight=0.3)
    g.add_edge("pred", "tgt", weight=0.8)
    merge_node_into(g, "src", "tgt")
    # Max of 0.3 and 0.8 is 0.8
    assert g["pred"]["tgt"]["weight"] == 0.8


# ---------------------------------------------------------------------------
# create_output_directory
# ---------------------------------------------------------------------------


def test_create_output_directory_creates_dir():
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = os.path.join(tmp, "new_subdir")
        assert not os.path.exists(new_dir)
        create_output_directory(new_dir)
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)


def test_create_output_directory_idempotent_when_exists():
    with tempfile.TemporaryDirectory() as tmp:
        # Should not raise when dir already exists
        create_output_directory(tmp)
        assert os.path.exists(tmp)


# ===========================================================================
# Pure config/logging functions
# ===========================================================================


import logging
import os
import textwrap
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# taxonomy.py
# ---------------------------------------------------------------------------


class TestEntityGroupConfig:
    def _make(self, tokens=None, patterns=None):
        from graphknows.ingestion.models.taxonomy import EntityGroupConfig

        return EntityGroupConfig(
            children=[],
            aliases=[],
            classification_tokens=tokens or [],
            classification_patterns=patterns or [],
        )

    def test_matches_name_by_token(self):
        eg = self._make(tokens=["COMMISSION"])
        assert eg.matches_name("EUROPEAN_COMMISSION") is True

    def test_matches_name_by_pattern(self):
        eg = self._make(patterns=[r"PARTY$"])
        assert eg.matches_name("GREEN_PARTY") is True

    def test_no_match(self):
        eg = self._make(tokens=["BANK"], patterns=[r"UNION$"])
        assert eg.matches_name("PARLIAMENT") is False

    def test_empty_group_always_false(self):
        eg = self._make()
        assert eg.matches_name("ANYTHING") is False


class TestTaxonomyConfig:
    def _make(self):
        from graphknows.ingestion.models.taxonomy import EntityGroupConfig, TaxonomyConfig

        return TaxonomyConfig(
            name="test",
            entity_groups={
                "PERSON": EntityGroupConfig(min_ner_confidence=0.7),
                "ORG": EntityGroupConfig(min_ner_confidence=0.5),
            },
        )

    def test_get_min_ner_confidence_known(self):
        tax = self._make()
        assert tax.get_min_ner_confidence("PERSON") == pytest.approx(0.7)

    def test_get_min_ner_confidence_unknown(self):
        tax = self._make()
        assert tax.get_min_ner_confidence("MISSING") == 0.0

    def test_entity_label_keys(self):
        tax = self._make()
        assert set(tax.entity_label_keys()) == {"PERSON", "ORG"}


class TestLoadTaxonomy:
    def test_load_taxonomy_missing_file(self, tmp_path):
        from graphknows.ingestion.models.taxonomy import load_taxonomy

        with pytest.raises(FileNotFoundError):
            load_taxonomy(str(tmp_path / "nonexistent.yaml"))

    def test_load_taxonomy_valid_yaml(self, tmp_path):
        from graphknows.ingestion.models.taxonomy import load_taxonomy

        yaml_content = textwrap.dedent(
            """\
            name: mytest
            version: "2.0"
            entity_groups:
              PERSON:
                children: []
                aliases: []
                min_ner_confidence: 0.8
            relation_types:
              - KNOWS
            """
        )
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        tax = load_taxonomy(str(p))
        assert tax.name == "mytest"
        assert tax.version == "2.0"
        assert "PERSON" in tax.entity_groups
        assert tax.relation_types == ["KNOWS"]


class TestGetDefaultTaxonomy:
    def test_loads_without_env(self, monkeypatch):
        from graphknows.ingestion.models.taxonomy import get_default_taxonomy

        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        tax = get_default_taxonomy()
        assert tax.name  # non-empty name

    def test_respects_env_var(self, monkeypatch, tmp_path):
        from graphknows.ingestion.models.taxonomy import get_default_taxonomy

        yaml_content = "name: custom\nentity_groups: {}\nrelation_types: []\n"
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(yaml_content)
        # env var is relative to package dir — use absolute path instead
        monkeypatch.setenv("GRAPHKNOWS_TAXONOMY_FILE", str(custom_yaml))

        # Patch the path resolution so it uses our tmp file directly
        with patch(
            "graphknows.ingestion.models.taxonomy.load_taxonomy",
            return_value=MagicMock(name="custom"),
        ) as mock_load:
            get_default_taxonomy()
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# loader.py
# ---------------------------------------------------------------------------
# llm.py
# ---------------------------------------------------------------------------


class TestExtractSecret:
    def test_none_container_returns_none(self):
        from graphknows.llm import _extract_secret

        assert _extract_secret(None, "api_key") is None

    def test_dict_with_key(self):
        from graphknows.llm import _extract_secret

        assert _extract_secret({"api_key": "sk-abc"}, "api_key") == "sk-abc"

    def test_dict_missing_key(self):
        from graphknows.llm import _extract_secret

        assert _extract_secret({"other": "x"}, "api_key") is None

    def test_pydantic_secret_value(self):
        from graphknows.llm import _extract_secret

        mock = MagicMock()
        mock.api_key = MagicMock()
        mock.api_key.get_secret_value.return_value = "real_key"
        result = _extract_secret(mock, "api_key")
        assert result == "real_key"

    def test_masked_secret_reads_from_env(self, monkeypatch):
        from graphknows.llm import _MASKED_SECRET, _extract_secret

        monkeypatch.setenv("API_KEY", "env_value")
        mock = MagicMock()
        mock.api_key = _MASKED_SECRET
        result = _extract_secret(mock, "api_key")
        assert result == "env_value"


class TestNormalizeOpenrouterModel:
    def test_adds_prefix(self):
        from graphknows.llm import normalize_openrouter_model

        assert normalize_openrouter_model("gpt-4") == "openrouter/gpt-4"

    def test_idempotent_when_already_prefixed(self):
        from graphknows.llm import normalize_openrouter_model

        assert normalize_openrouter_model("openrouter/gpt-4") == "openrouter/gpt-4"

    def test_none_returns_default(self):
        from graphknows.llm import _DEFAULT_MODEL, normalize_openrouter_model

        result = normalize_openrouter_model(None)
        assert result == f"openrouter/{_DEFAULT_MODEL}"
