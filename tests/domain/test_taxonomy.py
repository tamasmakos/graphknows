import ast
import importlib
from pathlib import Path

import pytest

from graphknows.domain.taxonomy import (
    EntityGroupConfig,
    TaxonomyConfig,
    get_default_taxonomy,
    load_taxonomy,
)


class TestEntityGroupConfig:
    def test_matches_name_token_hit(self):
        cfg = EntityGroupConfig(label="PERSON", classification_tokens=["person", "human"])
        assert cfg.matches_name("this is a person")

    def test_matches_name_pattern_hit(self):
        cfg = EntityGroupConfig(label="ORG", classification_patterns=[r"\bInc\b"])
        assert cfg.matches_name("Acme Inc")

    def test_matches_name_miss(self):
        cfg = EntityGroupConfig(label="ORG", classification_tokens=["company"])
        assert not cfg.matches_name("a random sentence")

    def test_min_ner_confidence_default(self):
        cfg = EntityGroupConfig(label="X")
        assert isinstance(cfg.min_ner_confidence, float)


class TestTaxonomyConfig:
    def test_entity_label_keys_exist(self):
        t = get_default_taxonomy()
        assert len(t.entity_groups) > 0

    def test_get_min_ner_confidence_unknown(self):
        t = get_default_taxonomy()
        assert t.get_min_ner_confidence("NONEXISTENT_LABEL_XYZ") == 0.0


class TestLoadTaxonomy:
    def test_load_taxonomy_valid(self, tmp_path):
        yaml_content = "name: test\nversion: '1.0'\nentity_groups:\n  PERSON:\n    ner_labels: [PER]\n"
        f = tmp_path / "test.yaml"
        f.write_text(yaml_content)
        t = load_taxonomy(str(f))
        assert t.name == "test"

    def test_load_taxonomy_missing_file(self):
        with pytest.raises((FileNotFoundError, Exception)):
            load_taxonomy("/nonexistent/path/to/file.yaml")


class TestGetDefaultTaxonomy:
    def test_returns_taxonomy_config(self):
        t = get_default_taxonomy()
        assert isinstance(t, TaxonomyConfig)
        assert t.name

    def test_env_var_selects_eu_politics(self, monkeypatch, tmp_path):
        # Just check env var path is accepted when valid
        src = (
            Path(__file__).parent.parent.parent / "graphknows" / "domain" / "taxonomy.eu-politics.yaml"
        )
        if src.exists():
            monkeypatch.setenv("GRAPHKNOWS_TAXONOMY_FILE", str(src))
            t = get_default_taxonomy()
            assert isinstance(t, TaxonomyConfig)


class TestImportBoundary:
    def test_domain_taxonomy_does_not_import_graphgen(self):
        import graphknows.domain.taxonomy as m

        src = Path(m.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "graphknows.graphgen" not in node.module, (
                        f"domain/taxonomy.py must not import from graphgen: {node.module}"
                    )
