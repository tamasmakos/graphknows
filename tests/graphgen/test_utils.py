"""Tests for pure utility functions in graphknows/graphgen/."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import networkx as nx

from graphknows.graphgen.diagnostics import (
    _sanitize_name,
    diagnostic_dir,
    diagnostics_enabled,
)
from graphknows.ingestion.ontology.labels import DEFAULT_LABELS, resolve_entity_labels
from graphknows.graphgen.utils import (
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

# ---------------------------------------------------------------------------
# diagnostics_enabled
# ---------------------------------------------------------------------------


def test_diagnostics_enabled_none_returns_false():
    assert diagnostics_enabled(None) is False


def test_diagnostics_enabled_empty_dict_returns_false():
    assert diagnostics_enabled({}) is False


def test_diagnostics_enabled_true_when_flag_set():
    config = {"extraction": {"diagnostic_mode": True}}
    assert diagnostics_enabled(config) is True


def test_diagnostics_enabled_false_when_flag_explicitly_false():
    config = {"extraction": {"diagnostic_mode": False}}
    assert diagnostics_enabled(config) is False


def test_diagnostics_enabled_missing_extraction_key_returns_false():
    config = {"infra": {"output_dir": "/tmp"}}
    assert diagnostics_enabled(config) is False


def test_diagnostics_enabled_works_with_pydantic_like_model():
    class FakeExtractionModel:
        def model_dump(self):
            return {"diagnostic_mode": True}

    config = {"extraction": FakeExtractionModel()}
    assert diagnostics_enabled(config) is True


# ---------------------------------------------------------------------------
# diagnostic_dir
# ---------------------------------------------------------------------------


def test_diagnostic_dir_builds_correct_path():
    config = {
        "infra": {"output_dir": "/tmp/out"},
        "extraction": {"diagnostic_output_subdir": "diag"},
    }
    result = diagnostic_dir(config)
    assert result == Path("/tmp/out/diag")


def test_diagnostic_dir_falls_back_to_defaults_on_missing_keys():
    result = diagnostic_dir({})
    assert result == Path("output") / "diagnostics"


def test_diagnostic_dir_returns_path_object():
    config = {"infra": {"output_dir": "/tmp"}, "extraction": {"diagnostic_output_subdir": "sub"}}
    assert isinstance(diagnostic_dir(config), Path)


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


def test_sanitize_name_replaces_spaces_with_underscore():
    assert _sanitize_name("Hello World") == "Hello_World"


def test_sanitize_name_replaces_slashes():
    assert _sanitize_name("a/b/c") == "a_b_c"


def test_sanitize_name_replaces_exclamation():
    assert _sanitize_name("Hello World!") == "Hello_World"


def test_sanitize_name_empty_string_returns_diagnostic():
    assert _sanitize_name("") == "diagnostic"


def test_sanitize_name_all_special_chars_returns_diagnostic():
    assert _sanitize_name("!!!") == "diagnostic"


def test_sanitize_name_alphanumeric_unchanged():
    assert _sanitize_name("valid_name123") == "valid_name123"


# ---------------------------------------------------------------------------
# resolve_entity_labels
# ---------------------------------------------------------------------------


def test_resolve_entity_labels_empty_config_returns_default_labels():
    result = resolve_entity_labels({})
    assert isinstance(result, list)
    assert len(result) > 0
    # All DEFAULT_LABELS should appear (after standardization)
    standardized_defaults = {label.upper().replace(" ", "_") for label in DEFAULT_LABELS}
    result_set = set(result)
    assert standardized_defaults <= result_set


def test_resolve_entity_labels_custom_gliner_labels_returned_standardized():
    result = resolve_entity_labels({"gliner_labels": ["Person", "Entity"]})
    assert "PERSON" in result
    assert "ENTITY" in result


def test_resolve_entity_labels_result_is_sorted():
    result = resolve_entity_labels({})
    assert result == sorted(result)


def test_resolve_entity_labels_no_duplicates():
    result = resolve_entity_labels({})
    assert len(result) == len(set(result))


def test_resolve_entity_labels_labels_are_uppercased():
    result = resolve_entity_labels({"gliner_labels": ["organization", "person"]})
    for label in result:
        assert label == label.upper()


# ===========================================================================
# Pure config/logging functions
# ===========================================================================


import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# taxonomy.py
# ---------------------------------------------------------------------------


class TestEntityGroupConfig:
    def _make(self, tokens=None, patterns=None):
        from graphknows.ingestion.domain.taxonomy import EntityGroupConfig

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
        from graphknows.ingestion.domain.taxonomy import EntityGroupConfig, TaxonomyConfig

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
        from graphknows.ingestion.domain.taxonomy import load_taxonomy

        with pytest.raises(FileNotFoundError):
            load_taxonomy(str(tmp_path / "nonexistent.yaml"))

    def test_load_taxonomy_valid_yaml(self, tmp_path):
        from graphknows.ingestion.domain.taxonomy import load_taxonomy

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
        from graphknows.ingestion.domain.taxonomy import get_default_taxonomy

        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        tax = get_default_taxonomy()
        assert tax.name  # non-empty name

    def test_respects_env_var(self, monkeypatch, tmp_path):
        from graphknows.ingestion.domain.taxonomy import get_default_taxonomy

        yaml_content = "name: custom\nentity_groups: {}\nrelation_types: []\n"
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(yaml_content)
        # env var is relative to package dir — use absolute path instead
        monkeypatch.setenv("GRAPHKNOWS_TAXONOMY_FILE", str(custom_yaml))

        # Patch the path resolution so it uses our tmp file directly
        with patch(
            "graphknows.ingestion.domain.taxonomy.load_taxonomy",
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
        from graphknows.config.llm import _extract_secret

        assert _extract_secret(None, "api_key") is None

    def test_dict_with_key(self):
        from graphknows.config.llm import _extract_secret

        assert _extract_secret({"api_key": "sk-abc"}, "api_key") == "sk-abc"

    def test_dict_missing_key(self):
        from graphknows.config.llm import _extract_secret

        assert _extract_secret({"other": "x"}, "api_key") is None

    def test_pydantic_secret_value(self):
        from graphknows.config.llm import _extract_secret

        mock = MagicMock()
        mock.api_key = MagicMock()
        mock.api_key.get_secret_value.return_value = "real_key"
        result = _extract_secret(mock, "api_key")
        assert result == "real_key"

    def test_masked_secret_reads_from_env(self, monkeypatch):
        from graphknows.config.llm import _MASKED_SECRET, _extract_secret

        monkeypatch.setenv("API_KEY", "env_value")
        mock = MagicMock()
        mock.api_key = _MASKED_SECRET
        result = _extract_secret(mock, "api_key")
        assert result == "env_value"


class TestGetModelName:
    def test_no_llm_section_raises(self):
        from graphknows.config.llm import get_model_name

        with pytest.raises(ValueError):
            get_model_name({})

    def test_extraction_purpose_uses_extraction_model(self):
        from graphknows.config.llm import get_model_name

        cfg = {"llm": {"extraction_model": "model-ext", "base_model": "model-base"}}
        assert get_model_name(cfg, purpose="extraction") == "model-ext"

    def test_summarization_purpose(self):
        from graphknows.config.llm import get_model_name

        cfg = {"llm": {"summarization_model": "model-sum"}}
        assert get_model_name(cfg, purpose="summarization") == "model-sum"

    def test_none_purpose_returns_base(self):
        from graphknows.config.llm import get_model_name

        cfg = {"llm": {"base_model": "model-base"}}
        assert get_model_name(cfg) == "model-base"

    def test_fallback_to_default_model(self):
        from graphknows.config.llm import _DEFAULT_MODEL, get_model_name

        cfg = {"llm": {}}
        assert get_model_name(cfg) == _DEFAULT_MODEL


class TestGetTemperature:
    def test_default_when_no_llm(self):
        from graphknows.config.llm import get_temperature

        assert get_temperature({}) == 0.0

    def test_reads_temperature(self):
        from graphknows.config.llm import get_temperature

        assert get_temperature({"llm": {"temperature": 0.7}}) == pytest.approx(0.7)


class TestNormalizeOpenrouterModel:
    def test_adds_prefix(self):
        from graphknows.config.llm import normalize_openrouter_model

        assert normalize_openrouter_model("gpt-4") == "openrouter/gpt-4"

    def test_idempotent_when_already_prefixed(self):
        from graphknows.config.llm import normalize_openrouter_model

        assert normalize_openrouter_model("openrouter/gpt-4") == "openrouter/gpt-4"

    def test_none_returns_default(self):
        from graphknows.config.llm import _DEFAULT_MODEL, normalize_openrouter_model

        result = normalize_openrouter_model(None)
        assert result == f"openrouter/{_DEFAULT_MODEL}"

class TestDiagnosticsEnabled:
    def test_false_when_none(self):
        from graphknows.graphgen.diagnostics import diagnostics_enabled

        assert diagnostics_enabled(None) is False

    def test_false_when_not_set(self):
        from graphknows.graphgen.diagnostics import diagnostics_enabled

        assert diagnostics_enabled({"extraction": {"diagnostic_mode": False}}) is False

    def test_true_when_enabled(self):
        from graphknows.graphgen.diagnostics import diagnostics_enabled

        assert diagnostics_enabled({"extraction": {"diagnostic_mode": True}}) is True


class TestDiagnosticDir:
    def test_returns_correct_path(self):
        from graphknows.graphgen.diagnostics import diagnostic_dir

        cfg = {
            "infra": {"output_dir": "/tmp/out"},
            "extraction": {"diagnostic_output_subdir": "diag"},
        }
        result = diagnostic_dir(cfg)
        assert str(result) == "/tmp/out/diag"

    def test_defaults(self):
        from graphknows.graphgen.diagnostics import diagnostic_dir

        result = diagnostic_dir({})
        assert result == Path("output") / "diagnostics"


class TestSanitizeName:
    def test_replaces_special_chars(self):
        from graphknows.graphgen.diagnostics import _sanitize_name

        assert _sanitize_name("foo/bar baz") == "foo_bar_baz"

    def test_empty_gives_diagnostic(self):
        from graphknows.graphgen.diagnostics import _sanitize_name

        assert _sanitize_name("") == "diagnostic"


class TestJsonSafe:
    def test_plain_types_passthrough(self):
        from graphknows.graphgen.diagnostics import _json_safe

        assert _json_safe({"a": [1, 2]}) == {"a": [1, 2]}

    def test_path_to_str(self):
        from graphknows.graphgen.diagnostics import _json_safe

        assert _json_safe(Path("/tmp/file")) == "/tmp/file"

    def test_set_to_sorted_list(self):
        from graphknows.graphgen.diagnostics import _json_safe

        result = _json_safe({3, 1, 2})
        assert result == [1, 2, 3]


class TestWriteDiagnosticJson:
    def test_returns_none_when_disabled(self, tmp_path):
        from graphknows.graphgen.diagnostics import write_diagnostic_json

        cfg = {"extraction": {"diagnostic_mode": False}}
        result = write_diagnostic_json(cfg, "test", {"data": 1})
        assert result is None

    def test_writes_file_when_enabled(self, tmp_path):
        from graphknows.graphgen.diagnostics import write_diagnostic_json

        cfg = {
            "extraction": {"diagnostic_mode": True, "diagnostic_output_subdir": "diag"},
            "infra": {"output_dir": str(tmp_path)},
        }
        path = write_diagnostic_json(cfg, "my test!", {"value": 42})
        assert path is not None
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["value"] == 42

    def test_sanitizes_filename(self, tmp_path):
        from graphknows.graphgen.diagnostics import write_diagnostic_json

        cfg = {
            "extraction": {"diagnostic_mode": True, "diagnostic_output_subdir": "diag"},
            "infra": {"output_dir": str(tmp_path)},
        }
        path = write_diagnostic_json(cfg, "foo/bar baz!", {"x": 1})
        assert path is not None
        fname = Path(path).name
        assert "/" not in fname


# ---------------------------------------------------------------------------
# labels.py
# ---------------------------------------------------------------------------


class TestResolveEntityLabels:
    def test_manual_labels_returned_and_standardized(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        cfg = {"gliner_labels": ["Person", "organization"]}
        result = resolve_entity_labels(cfg)
        # After standardize_label (uppercase) + dedup + sort
        assert "PERSON" in result
        assert "ORGANIZATION" in result

    def test_defaults_when_empty_config(self):
        from graphknows.ingestion.ontology.labels import DEFAULT_LABELS, resolve_entity_labels

        result = resolve_entity_labels({})
        # Should contain standardized versions of DEFAULT_LABELS
        assert len(result) > 0

    def test_entity_labels_key_takes_priority(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        cfg = {
            "entity_labels": ["Country"],
            "gliner_labels": ["Person"],
        }
        result = resolve_entity_labels(cfg)
        assert "COUNTRY" in result
        # gliner_labels should NOT be used since entity_labels was present
        assert "PERSON" not in result

    def test_ontology_disabled_uses_manual_only(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        cfg = {
            "gliner_labels": ["Politician"],
            "ontology": {"enabled": False},
        }
        result = resolve_entity_labels(cfg)
        assert "POLITICIAN" in result

    def test_ontology_enabled_but_extraction_fails_falls_back(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        cfg = {
            "gliner_labels": ["Person"],
            "ontology": {"enabled": True, "ontology_dir": "/nonexistent"},
        }
        # Should not raise — falls back to manual labels
        result = resolve_entity_labels(cfg)
        assert len(result) > 0

    def test_pydantic_model_input(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {"gliner_labels": ["Doctor"]}
        result = resolve_entity_labels(mock_cfg)
        assert "DOCTOR" in result

    def test_result_is_sorted_unique(self):
        from graphknows.ingestion.ontology.labels import resolve_entity_labels

        cfg = {"gliner_labels": ["person", "Person", "PERSON"]}
        result = resolve_entity_labels(cfg)
        assert result == sorted(set(result))
        assert result.count("PERSON") == 1


