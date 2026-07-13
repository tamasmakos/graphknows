"""Unit tests for the conversational taxonomy and related components.

Covers:
- taxonomy.conversational.yaml parses correctly
- get_default_taxonomy() returns the conversational taxonomy
- Required entity groups are present (PERSON, PREFERENCE, RELATIONSHIP, etc.)
- Required relation predicates are present
- label_space derivation from taxonomy (no hardcoded political labels)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphknows.ingestion.models.taxonomy import (
    TaxonomyConfig,
    get_default_taxonomy,
    load_taxonomy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YAML_PATH = (
    Path(__file__).parent.parent.parent
    / "graphknows"
    / "ingestion"
    / "models"
    / "taxonomy.conversational.yaml"
)

_REQUIRED_GROUPS = {
    "PERSON",
    "RELATIONSHIP",
    "EVENT",
    "ACTIVITY",
    "PREFERENCE",
    "POSSESSION",
    "LOCATION",
    "ORGANIZATION",
    "TEMPORAL",
    "EMOTION",
    "ATTRIBUTE",
    "GOAL",
}

_REQUIRED_RELATIONS = {
    "lives_in",
    "works_as",
    "friend_of",
    "family_of",
    "partner_of",
    "attended",
    "owns",
    "likes",
    "dislikes",
    "plans_to",
    "happened_on",
    "located_in",
    "member_of",
}


# ---------------------------------------------------------------------------
# YAML file existence and basic parse
# ---------------------------------------------------------------------------


class TestConversationalYamlExists:
    def test_yaml_file_exists(self):
        assert _YAML_PATH.exists(), f"taxonomy.conversational.yaml not found at {_YAML_PATH}"

    def test_yaml_parses_to_taxonomy_config(self):
        t = load_taxonomy(str(_YAML_PATH))
        assert isinstance(t, TaxonomyConfig)

    def test_yaml_name_is_conversational(self):
        t = load_taxonomy(str(_YAML_PATH))
        assert t.name == "conversational"

    def test_yaml_has_version(self):
        t = load_taxonomy(str(_YAML_PATH))
        assert t.version


# ---------------------------------------------------------------------------
# Entity groups
# ---------------------------------------------------------------------------


class TestConversationalEntityGroups:
    @pytest.fixture(scope="class")
    def taxonomy(self):
        return load_taxonomy(str(_YAML_PATH))

    def test_all_required_groups_present(self, taxonomy):
        keys = set(taxonomy.entity_label_keys())
        missing = _REQUIRED_GROUPS - keys
        assert not missing, f"Missing required entity groups: {missing}"

    def test_person_group_has_aliases(self, taxonomy):
        person = taxonomy.entity_groups["PERSON"]
        assert len(person.aliases) > 0

    def test_preference_group_has_aliases(self, taxonomy):
        pref = taxonomy.entity_groups["PREFERENCE"]
        assert any(a in pref.aliases for a in ("like", "love", "hate", "prefer", "favourite", "favorite"))

    def test_relationship_group_has_kinship_aliases(self, taxonomy):
        rel = taxonomy.entity_groups["RELATIONSHIP"]
        assert any(a in rel.aliases for a in ("sister", "brother", "mother", "father", "friend"))

    def test_temporal_group_has_aliases(self, taxonomy):
        temporal = taxonomy.entity_groups["TEMPORAL"]
        assert len(temporal.aliases) > 0

    def test_all_groups_have_descriptions(self, taxonomy):
        for key, group in taxonomy.entity_groups.items():
            assert group.description.strip(), f"Group {key!r} has no description"


# ---------------------------------------------------------------------------
# Relation types
# ---------------------------------------------------------------------------


class TestConversationalRelationTypes:
    @pytest.fixture(scope="class")
    def taxonomy(self):
        return load_taxonomy(str(_YAML_PATH))

    def test_all_required_relations_present(self, taxonomy):
        present = set(taxonomy.relation_types)
        missing = _REQUIRED_RELATIONS - present
        assert not missing, f"Missing required relation types: {missing}"

    def test_relation_count(self, taxonomy):
        # Must have at least all required relations
        assert len(taxonomy.relation_types) >= len(_REQUIRED_RELATIONS)


# ---------------------------------------------------------------------------
# get_default_taxonomy() returns conversational
# ---------------------------------------------------------------------------


class TestGetDefaultTaxonomyIsConversational:
    def test_default_is_conversational(self, monkeypatch):
        # Ensure env var is not set so we get the true default
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        assert isinstance(t, TaxonomyConfig)
        assert t.name == "conversational", (
            f"Expected default taxonomy name 'conversational', got {t.name!r}"
        )

    def test_default_contains_person_group(self, monkeypatch):
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        assert "PERSON" in t.entity_groups

    def test_default_contains_preference_group(self, monkeypatch):
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        assert "PREFERENCE" in t.entity_groups

    def test_default_contains_relationship_group(self, monkeypatch):
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        assert "RELATIONSHIP" in t.entity_groups

    def test_default_relation_predicates(self, monkeypatch):
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        for rel in ("likes", "dislikes", "friend_of", "family_of", "lives_in"):
            assert rel in t.relation_types, f"Relation {rel!r} missing from default taxonomy"

    def test_entity_label_keys_sorted_output(self, monkeypatch):
        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        t = get_default_taxonomy()
        keys = t.entity_label_keys()
        assert isinstance(keys, list)
        assert len(keys) >= len(_REQUIRED_GROUPS)


# ---------------------------------------------------------------------------
# env var override still works
# ---------------------------------------------------------------------------


class TestEnvVarOverride:
    def test_env_var_selects_intentionality(self, monkeypatch):
        intentionality_path = (
            Path(__file__).parent.parent.parent
            / "graphknows"
            / "ingestion"
            / "domain"
            / "taxonomy.intentionality.yaml"
        )
        if not intentionality_path.exists():
            pytest.skip("taxonomy.intentionality.yaml not present")
        monkeypatch.setenv("GRAPHKNOWS_TAXONOMY_FILE", str(intentionality_path))
        t = get_default_taxonomy()
        assert t.name == "intentionality"


# ---------------------------------------------------------------------------
# label_space derivation from conversational taxonomy
# ---------------------------------------------------------------------------


class TestLabelSpaceFromTaxonomy:
    """Ensure build_top_level_label_space works with the conversational taxonomy.

    These tests import label_space module-level constants and the public
    build_top_level_label_space function directly, bypassing the heavy ML deps
    that the graphknows.extraction package __init__.py eagerly loads.

    Skipped automatically when the extraction package cannot be imported (e.g.
    langchain_text_splitters not present in the test environment).
    """

    @pytest.fixture(scope="class", autouse=True)
    def require_extraction(self):
        """Skip the whole class if extraction deps are unavailable."""
        pytest.importorskip("langchain_text_splitters", reason="ML deps not installed")

    def test_label_space_derived_from_taxonomy(self, monkeypatch):
        from graphknows.ingestion.extraction.entities.label_space import build_top_level_label_space

        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        taxonomy = get_default_taxonomy()
        ontology_labels = taxonomy.entity_label_keys()
        label_space = build_top_level_label_space(ontology_labels, taxonomy=taxonomy)
        assert "PERSON" in label_space
        assert "PREFERENCE" in label_space
        assert "RELATIONSHIP" in label_space

    def test_label_space_aliases_come_from_taxonomy(self, monkeypatch):
        from graphknows.ingestion.extraction.entities.label_space import build_top_level_label_space

        monkeypatch.delenv("GRAPHKNOWS_TAXONOMY_FILE", raising=False)
        taxonomy = get_default_taxonomy()
        ontology_labels = taxonomy.entity_label_keys()
        label_space = build_top_level_label_space(ontology_labels, taxonomy=taxonomy)
        person_aliases = label_space.get("PERSON", {}).get("aliases", [])
        # Aliases should come from the YAML, not from hardcoded political labels
        assert "person" in person_aliases
        # Political-domain specific aliases should not appear
        political_aliases = {"commissioner", "kremlin", "parliament"}
        assert not political_aliases.intersection(set(person_aliases)), (
            f"Political aliases leaked into PERSON label space: "
            f"{political_aliases.intersection(set(person_aliases))}"
        )
