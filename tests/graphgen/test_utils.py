"""Tests for pure utility functions in graphknows/graphgen/utils/."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import networkx as nx

from graphknows.graphgen.utils.diagnostics import (
    _sanitize_name,
    diagnostic_dir,
    diagnostics_enabled,
)
from graphknows.graphgen.utils.labels import DEFAULT_LABELS, resolve_entity_labels
from graphknows.graphgen.utils.provenance import get_git_revision, redact_secrets
from graphknows.graphgen.utils.utils import (
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
# redact_secrets
# ---------------------------------------------------------------------------


def test_redact_secrets_api_key_is_redacted():
    data = {"api_key": "SECRET123", "name": "Alice"}
    result = redact_secrets(data)
    assert result["api_key"] == "REDACTED"
    assert result["name"] == "Alice"


def test_redact_secrets_non_sensitive_keys_unchanged():
    data = {"username": "alice", "email": "alice@example.com"}
    result = redact_secrets(data)
    assert result["username"] == "alice"
    assert result["email"] == "alice@example.com"


def test_redact_secrets_nested_dict():
    data = {"config": {"token": "abc123", "host": "localhost"}}
    result = redact_secrets(data)
    assert result["config"]["token"] == "REDACTED"
    assert result["config"]["host"] == "localhost"


def test_redact_secrets_list_of_dicts():
    data = [{"api_key": "x"}, {"name": "Bob"}]
    result = redact_secrets(data)
    assert result[0]["api_key"] == "REDACTED"
    assert result[1]["name"] == "Bob"


def test_redact_secrets_case_insensitive_key_matching():
    # "password", "API_KEY", "Token", "SECRET_KEY" all contain secret keywords
    data = {
        "Password": "hunter2",
        "API_KEY": "key-val",
        "Token": "tok123",
        "SECRET_KEY": "shhh",
    }
    result = redact_secrets(data)
    assert result["Password"] == "REDACTED"
    assert result["API_KEY"] == "REDACTED"
    assert result["Token"] == "REDACTED"
    assert result["SECRET_KEY"] == "REDACTED"


def test_redact_secrets_empty_dict_returns_empty_dict():
    assert redact_secrets({}) == {}


def test_redact_secrets_empty_list_returns_empty_list():
    assert redact_secrets([]) == []


# ---------------------------------------------------------------------------
# get_git_revision
# ---------------------------------------------------------------------------


def test_get_git_revision_real_repo_returns_string():
    result = get_git_revision("/root/graphknows")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_git_revision_nonexistent_dir_returns_none():
    result = get_git_revision("/nonexistent/path/abc123")
    assert result is None


def test_get_git_revision_dir_without_git_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        result = get_git_revision(tmp)
        assert result is None


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
    standardized_defaults = {
        label.upper().replace(" ", "_") for label in DEFAULT_LABELS
    }
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
        from graphknows.graphgen.config.taxonomy import EntityGroupConfig

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
        from graphknows.graphgen.config.taxonomy import EntityGroupConfig, TaxonomyConfig

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
        from graphknows.graphgen.config.taxonomy import load_taxonomy

        with pytest.raises(FileNotFoundError):
            load_taxonomy(str(tmp_path / "nonexistent.yaml"))

    def test_load_taxonomy_valid_yaml(self, tmp_path):
        from graphknows.graphgen.config.taxonomy import load_taxonomy

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
        from graphknows.graphgen.config.taxonomy import get_default_taxonomy

        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        tax = get_default_taxonomy()
        assert tax.name  # non-empty name

    def test_respects_env_var(self, monkeypatch, tmp_path):
        from graphknows.graphgen.config.taxonomy import get_default_taxonomy

        yaml_content = "name: custom\nentity_groups: {}\nrelation_types: []\n"
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(yaml_content)
        # env var is relative to package dir — use absolute path instead
        monkeypatch.setenv("GRAPHKNOWS_TAXONOMY_FILE", str(custom_yaml))

        # Patch the path resolution so it uses our tmp file directly
        with patch(
            "graphknows.graphgen.config.taxonomy.load_taxonomy",
            return_value=MagicMock(name="custom"),
        ) as mock_load:
            get_default_taxonomy()
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# loader.py
# ---------------------------------------------------------------------------


class TestLoadYamlConfig:
    def test_returns_empty_dict_when_file_missing(self):
        from graphknows.graphgen.config.loader import load_yaml_config

        result = load_yaml_config("/nonexistent/config.yaml")
        assert result == {}

    def test_loads_valid_yaml(self, tmp_path):
        from graphknows.graphgen.config.loader import load_yaml_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("key: value\nnumber: 42\n")
        result = load_yaml_config(str(cfg_file))
        assert result == {"key": "value", "number": 42}

    def test_returns_empty_on_broken_yaml(self, tmp_path):
        from graphknows.graphgen.config.loader import load_yaml_config

        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [\n")  # malformed
        result = load_yaml_config(str(bad))
        assert result == {}

    def test_returns_empty_on_empty_file(self, tmp_path):
        from graphknows.graphgen.config.loader import load_yaml_config

        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        result = load_yaml_config(str(empty))
        assert result == {}


# ---------------------------------------------------------------------------
# llm.py
# ---------------------------------------------------------------------------


class TestExtractSecret:
    def test_none_container_returns_none(self):
        from graphknows.graphgen.config.llm import _extract_secret

        assert _extract_secret(None, "api_key") is None

    def test_dict_with_key(self):
        from graphknows.graphgen.config.llm import _extract_secret

        assert _extract_secret({"api_key": "sk-abc"}, "api_key") == "sk-abc"

    def test_dict_missing_key(self):
        from graphknows.graphgen.config.llm import _extract_secret

        assert _extract_secret({"other": "x"}, "api_key") is None

    def test_pydantic_secret_value(self):
        from graphknows.graphgen.config.llm import _extract_secret

        mock = MagicMock()
        mock.api_key = MagicMock()
        mock.api_key.get_secret_value.return_value = "real_key"
        result = _extract_secret(mock, "api_key")
        assert result == "real_key"

    def test_masked_secret_reads_from_env(self, monkeypatch):
        from graphknows.graphgen.config.llm import _MASKED_SECRET, _extract_secret

        monkeypatch.setenv("API_KEY", "env_value")
        mock = MagicMock()
        mock.api_key = _MASKED_SECRET
        result = _extract_secret(mock, "api_key")
        assert result == "env_value"


class TestGetModelName:
    def test_no_llm_section_raises(self):
        from graphknows.graphgen.config.llm import get_model_name

        with pytest.raises(ValueError):
            get_model_name({})

    def test_extraction_purpose_uses_extraction_model(self):
        from graphknows.graphgen.config.llm import get_model_name

        cfg = {"llm": {"extraction_model": "model-ext", "base_model": "model-base"}}
        assert get_model_name(cfg, purpose="extraction") == "model-ext"

    def test_summarization_purpose(self):
        from graphknows.graphgen.config.llm import get_model_name

        cfg = {"llm": {"summarization_model": "model-sum"}}
        assert get_model_name(cfg, purpose="summarization") == "model-sum"

    def test_none_purpose_returns_base(self):
        from graphknows.graphgen.config.llm import get_model_name

        cfg = {"llm": {"base_model": "model-base"}}
        assert get_model_name(cfg) == "model-base"

    def test_fallback_to_default_model(self):
        from graphknows.graphgen.config.llm import _DEFAULT_MODEL, get_model_name

        cfg = {"llm": {}}
        assert get_model_name(cfg) == _DEFAULT_MODEL


class TestGetTemperature:
    def test_default_when_no_llm(self):
        from graphknows.graphgen.config.llm import get_temperature

        assert get_temperature({}) == 0.0

    def test_reads_temperature(self):
        from graphknows.graphgen.config.llm import get_temperature

        assert get_temperature({"llm": {"temperature": 0.7}}) == pytest.approx(0.7)


class TestNormalizeOpenrouterModel:
    def test_adds_prefix(self):
        from graphknows.graphgen.config.llm import normalize_openrouter_model

        assert normalize_openrouter_model("gpt-4") == "openrouter/gpt-4"

    def test_idempotent_when_already_prefixed(self):
        from graphknows.graphgen.config.llm import normalize_openrouter_model

        assert normalize_openrouter_model("openrouter/gpt-4") == "openrouter/gpt-4"

    def test_none_returns_default(self):
        from graphknows.graphgen.config.llm import _DEFAULT_MODEL, normalize_openrouter_model

        result = normalize_openrouter_model(None)
        assert result == f"openrouter/{_DEFAULT_MODEL}"


# ---------------------------------------------------------------------------
# utils/logging.py
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_configure_sets_debug_level(self):
        from graphknows.graphgen.utils.logging import configure_logging

        configure_logging(debug=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_default_is_info(self):
        from graphknows.graphgen.utils.logging import configure_logging

        configure_logging(debug=False)
        assert logging.getLogger().level == logging.INFO

    def test_configure_explicit_level(self):
        from graphknows.graphgen.utils.logging import configure_logging

        configure_logging(level=logging.WARNING)
        assert logging.getLogger().level == logging.WARNING


# ---------------------------------------------------------------------------
# provenance.py
# ---------------------------------------------------------------------------


class TestIsSecretKey:
    @pytest.mark.parametrize("key", ["api_key", "my_token", "client_secret", "secret_key"])
    def test_detects_secret_keywords(self, key):
        from graphknows.graphgen.utils.provenance import _is_secret_key

        assert _is_secret_key(key) is True

    def test_non_secret_key(self):
        from graphknows.graphgen.utils.provenance import _is_secret_key

        assert _is_secret_key("username") is False


class TestRedactSecrets:
    def test_redacts_api_key_in_dict(self):
        from graphknows.graphgen.utils.provenance import redact_secrets

        data = {"api_key": "super-secret", "name": "test"}
        result = redact_secrets(data)
        assert result["api_key"] == "REDACTED"
        assert result["name"] == "test"

    def test_recursive_redaction(self):
        from graphknows.graphgen.utils.provenance import redact_secrets

        data = {"outer": {"token": "abc", "value": 42}}
        result = redact_secrets(data)
        assert result["outer"]["token"] == "REDACTED"
        assert result["outer"]["value"] == 42

    def test_list_passthrough(self):
        from graphknows.graphgen.utils.provenance import redact_secrets

        result = redact_secrets(["a", "b"])
        assert result == ["a", "b"]

    def test_pydantic_secret_type_redacted(self):
        from graphknows.graphgen.utils.provenance import redact_secrets

        secret = MagicMock()
        secret.get_secret_value = MagicMock(return_value="real")
        result = redact_secrets({"key": secret})
        # _redact picks it up — but key isn't secret → value is run through _redact
        # The mock has get_secret_value so _redact returns "REDACTED"
        assert result["key"] == "REDACTED"


class TestSafeSerialize:
    def test_json_serializable_passthrough(self):
        from graphknows.graphgen.utils.provenance import _safe_serialize

        obj = {"a": [1, 2, 3], "b": "hello"}
        assert _safe_serialize(obj) == obj

    def test_non_serializable_converted_to_str(self):
        from graphknows.graphgen.utils.provenance import _safe_serialize

        class _Unserializable:
            def __repr__(self):
                return "Unserializable()"

        result = _safe_serialize({"obj": _Unserializable()})
        assert isinstance(result["obj"], str)


class TestGetGitRevision:
    def test_returns_none_when_no_git_dir(self, tmp_path):
        from graphknows.graphgen.utils.provenance import get_git_revision

        result = get_git_revision(str(tmp_path))
        assert result is None

    def test_reads_direct_sha(self, tmp_path):
        from graphknows.graphgen.utils.provenance import get_git_revision

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("abc123def456\n", encoding="utf-8")
        result = get_git_revision(str(tmp_path))
        assert result == "abc123def456"

    def test_reads_symbolic_ref(self, tmp_path):
        from graphknows.graphgen.utils.provenance import get_git_revision

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        ref_dir = git_dir / "refs" / "heads"
        ref_dir.mkdir(parents=True)
        (ref_dir / "main").write_text("deadbeef\n", encoding="utf-8")
        result = get_git_revision(str(tmp_path))
        assert result == "deadbeef"


class TestGetPackageVersions:
    def test_returns_version_for_known_package(self):
        from graphknows.graphgen.utils.provenance import _get_package_versions

        result = _get_package_versions(["pydantic"])
        assert "pydantic" in result
        assert result["pydantic"] is not None

    def test_returns_none_for_unknown_package(self):
        from graphknows.graphgen.utils.provenance import _get_package_versions

        result = _get_package_versions(["__nonexistent_package_xyz__"])
        assert result["__nonexistent_package_xyz__"] is None


class TestCollectEnvironmentInfo:
    def test_returns_expected_keys(self):
        from graphknows.graphgen.utils.provenance import collect_environment_info

        info = collect_environment_info()
        assert "python_version" in info
        assert "platform" in info
        assert "package_versions" in info
        assert "pydantic" in info["package_versions"]


class TestWritePipelineConfigSnapshot:
    def test_writes_json_and_redacts(self, tmp_path):
        from graphknows.graphgen.utils.provenance import write_pipeline_config_snapshot

        class _FakeSettings:
            def model_dump(self) -> dict:
                return {"api_key": "secret", "model": "gpt-4"}

        path = write_pipeline_config_snapshot(str(tmp_path), _FakeSettings())
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["api_key"] == "REDACTED"
        assert data["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------


class TestDiagnosticsEnabled:
    def test_false_when_none(self):
        from graphknows.graphgen.utils.diagnostics import diagnostics_enabled

        assert diagnostics_enabled(None) is False

    def test_false_when_not_set(self):
        from graphknows.graphgen.utils.diagnostics import diagnostics_enabled

        assert diagnostics_enabled({"extraction": {"diagnostic_mode": False}}) is False

    def test_true_when_enabled(self):
        from graphknows.graphgen.utils.diagnostics import diagnostics_enabled

        assert diagnostics_enabled({"extraction": {"diagnostic_mode": True}}) is True


class TestDiagnosticDir:
    def test_returns_correct_path(self):
        from graphknows.graphgen.utils.diagnostics import diagnostic_dir

        cfg = {"infra": {"output_dir": "/tmp/out"}, "extraction": {"diagnostic_output_subdir": "diag"}}
        result = diagnostic_dir(cfg)
        assert str(result) == "/tmp/out/diag"

    def test_defaults(self):
        from graphknows.graphgen.utils.diagnostics import diagnostic_dir

        result = diagnostic_dir({})
        assert result == Path("output") / "diagnostics"


class TestSanitizeName:
    def test_replaces_special_chars(self):
        from graphknows.graphgen.utils.diagnostics import _sanitize_name

        assert _sanitize_name("foo/bar baz") == "foo_bar_baz"

    def test_empty_gives_diagnostic(self):
        from graphknows.graphgen.utils.diagnostics import _sanitize_name

        assert _sanitize_name("") == "diagnostic"


class TestJsonSafe:
    def test_plain_types_passthrough(self):
        from graphknows.graphgen.utils.diagnostics import _json_safe

        assert _json_safe({"a": [1, 2]}) == {"a": [1, 2]}

    def test_path_to_str(self):
        from graphknows.graphgen.utils.diagnostics import _json_safe

        assert _json_safe(Path("/tmp/file")) == "/tmp/file"

    def test_set_to_sorted_list(self):
        from graphknows.graphgen.utils.diagnostics import _json_safe

        result = _json_safe({3, 1, 2})
        assert result == [1, 2, 3]


class TestWriteDiagnosticJson:
    def test_returns_none_when_disabled(self, tmp_path):
        from graphknows.graphgen.utils.diagnostics import write_diagnostic_json

        cfg = {"extraction": {"diagnostic_mode": False}}
        result = write_diagnostic_json(cfg, "test", {"data": 1})
        assert result is None

    def test_writes_file_when_enabled(self, tmp_path):
        from graphknows.graphgen.utils.diagnostics import write_diagnostic_json

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
        from graphknows.graphgen.utils.diagnostics import write_diagnostic_json

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
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        cfg = {"gliner_labels": ["Person", "organization"]}
        result = resolve_entity_labels(cfg)
        # After standardize_label (uppercase) + dedup + sort
        assert "PERSON" in result
        assert "ORGANIZATION" in result

    def test_defaults_when_empty_config(self):
        from graphknows.graphgen.utils.labels import DEFAULT_LABELS, resolve_entity_labels

        result = resolve_entity_labels({})
        # Should contain standardized versions of DEFAULT_LABELS
        assert len(result) > 0

    def test_entity_labels_key_takes_priority(self):
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        cfg = {
            "entity_labels": ["Country"],
            "gliner_labels": ["Person"],
        }
        result = resolve_entity_labels(cfg)
        assert "COUNTRY" in result
        # gliner_labels should NOT be used since entity_labels was present
        assert "PERSON" not in result

    def test_ontology_disabled_uses_manual_only(self):
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        cfg = {
            "gliner_labels": ["Politician"],
            "ontology": {"enabled": False},
        }
        result = resolve_entity_labels(cfg)
        assert "POLITICIAN" in result

    def test_ontology_enabled_but_extraction_fails_falls_back(self):
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        cfg = {
            "gliner_labels": ["Person"],
            "ontology": {"enabled": True, "ontology_dir": "/nonexistent"},
        }
        # Should not raise — falls back to manual labels
        result = resolve_entity_labels(cfg)
        assert len(result) > 0

    def test_pydantic_model_input(self):
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {"gliner_labels": ["Doctor"]}
        result = resolve_entity_labels(mock_cfg)
        assert "DOCTOR" in result

    def test_result_is_sorted_unique(self):
        from graphknows.graphgen.utils.labels import resolve_entity_labels

        cfg = {"gliner_labels": ["person", "Person", "PERSON"]}
        result = resolve_entity_labels(cfg)
        assert result == sorted(set(result))
        assert result.count("PERSON") == 1


# ---------------------------------------------------------------------------
# patch/report.py
# ---------------------------------------------------------------------------


class TestFormatDistribution:
    def test_empty_returns_none_str(self):
        from graphknows.graphgen.patch.report import _format_distribution

        assert _format_distribution({}) == "none"

    def test_single_entry(self):
        from graphknows.graphgen.patch.report import _format_distribution

        assert _format_distribution({"PERSON": 5}) == "PERSON=5"

    def test_sorted_output(self):
        from graphknows.graphgen.patch.report import _format_distribution

        result = _format_distribution({"Z": 1, "A": 2})
        assert result == "A=2, Z=1"


class TestFormatPatchReport:
    def test_minimal_summary(self):
        from graphknows.graphgen.patch.report import format_patch_report

        summary = {
            "dry_run": False,
            "batch_size": 10,
            "phases": [],
            "totals": {"updated": 0, "before": 0, "after": 0},
        }
        result = format_patch_report(summary)
        assert "dry_run=False" in result
        assert "batch_size=10" in result
        assert "Totals:" in result

    def test_dry_run_label(self):
        from graphknows.graphgen.patch.report import format_patch_report

        summary = {
            "dry_run": True,
            "batch_size": 5,
            "phases": [
                {
                    "phase": "structural",
                    "before": 10,
                    "after": 8,
                    "updated": 2,
                    "ontology_class_distribution": {"PERSON": 3},
                }
            ],
            "totals": {"updated": 2, "before": 10, "after": 8},
        }
        result = format_patch_report(summary)
        assert "would update" in result
        assert "structural" in result
        assert "PERSON=3" in result

    def test_missing_distribution_handled(self):
        from graphknows.graphgen.patch.report import format_patch_report

        summary = {
            "dry_run": False,
            "batch_size": 1,
            "phases": [{"phase": "legacy", "updated": 0, "ontology_class_distribution": None}],
            "totals": {},
        }
        result = format_patch_report(summary)
        assert "legacy" in result
        assert "none" in result


# ---------------------------------------------------------------------------
# summarization/models.py
# ---------------------------------------------------------------------------


class TestSummarizationTask:
    def test_creates_with_required_fields(self):
        from graphknows.graphgen.pipeline.summarization.models import SummarizationTask

        task = SummarizationTask(
            task_id="t1",
            community_id=42,
            subcommunity_id=None,
            is_topic=True,
            chunk_texts=["hello world"],
            entities=[{"name": "Alice", "id": "e1", "degree": 3}],
            relationships=[("Alice", "KNOWS", "Bob")],
            chunk_ids=["c1"],
            entity_ids=["e1"],
        )
        assert task.task_id == "t1"
        assert task.community_id == 42
        assert task.is_topic is True
        assert task.title is None
        assert task.summary is None
        assert task.sub_summaries is None

    def test_can_set_output_fields(self):
        from graphknows.graphgen.pipeline.summarization.models import SummarizationTask

        task = SummarizationTask(
            task_id="t2",
            community_id=1,
            subcommunity_id=2,
            is_topic=False,
            chunk_texts=[],
            entities=[],
            relationships=[],
            chunk_ids=[],
            entity_ids=[],
            title="Test Title",
            summary="Test summary",
        )
        assert task.title == "Test Title"
        assert task.summary == "Test summary"


# ---------------------------------------------------------------------------
# graphgen/ingest.py — empty-document guard only (no DB needed)
# ---------------------------------------------------------------------------


class TestRunIngestionGuard:
    def test_empty_documents_returns_skipped(self):
        import asyncio

        from graphknows.graphgen.ingest import run_ingestion

        result = asyncio.run(run_ingestion(documents=[]))
        assert result["status"] == "skipped"
        assert "no documents" in result["reason"]
