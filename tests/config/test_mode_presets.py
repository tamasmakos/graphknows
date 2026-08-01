"""MemoryMode is a preset over orthogonal knobs, not a global fork.

Pins the expansion table and the override precedence, so the "resolved
capability" contract every consumer relies on cannot drift silently.
"""

from __future__ import annotations

import pytest

from graphknows.settings import (
    GraphKnowsSettings,
    MemoryMode,
    TopicMode,
)


class TestPresetExpansion:
    def test_llm_free_preset(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert s.topic_mode is TopicMode.none
        assert s.dspy_relations is False
        assert s.uses_llm is False

    def test_llm_assisted_preset(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted)
        assert s.topic_mode is TopicMode.llm
        assert s.dspy_relations is True
        assert s.uses_llm is True

    def test_default_matches_llm_free(self) -> None:
        """A bare construction must behave exactly like today's default."""
        s = GraphKnowsSettings()
        assert s.mode is MemoryMode.llm_free
        assert s.topic_mode is TopicMode.none
        assert s.dspy_relations is False


class TestKnobOverrides:
    def test_topics_opt_in_under_llm_free(self) -> None:
        """The headline case: deterministic topics without leaving llm_free."""
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
            GRAPHKNOWS_TOPICS=TopicMode.extractive,
        )
        assert s.topic_mode is TopicMode.extractive
        assert s.dspy_relations is False  # untouched by the override
        assert s.uses_llm is False  # extractive topics make no network call

    def test_dspy_disabled_under_llm_assisted(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_DSPY_RELATIONS=False,
        )
        assert s.dspy_relations is False
        assert s.topic_mode is TopicMode.llm  # other knobs keep the preset

    def test_every_knob_overridden_ignores_the_preset(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_TOPICS=TopicMode.none,
            GRAPHKNOWS_DSPY_RELATIONS=False,
        )
        assert s.uses_llm is False


class TestEmptyEnvVars:
    """An empty env var means "unset", not "invalid".

    docker-compose and shell wrappers spell an absent value as ``VAR=``; without
    this the empty string reaches pydantic and fails enum validation, taking the
    process down at construction.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"GRAPHKNOWS_TOPICS": ""},
            {"GRAPHKNOWS_DSPY_RELATIONS": ""},
            {"GRAPHKNOWS_TOPICS": "   "},
        ],
    )
    def test_empty_falls_back_to_the_preset(self, kwargs: dict[str, str]) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_free, **kwargs)
        assert s.topic_mode is TopicMode.none
        assert s.dspy_relations is False


class TestOntologySource:
    def test_unset_resolves_to_the_bundled_schemaorg_digest(self) -> None:
        """Ontology injection is always on: no configuration still yields one."""
        from pathlib import Path

        source = Path(GraphKnowsSettings().ontology_source)
        assert source.parts[-3:] == ("assets", "schemaorg", "schemaorg.json")
        assert source.is_file()

    def test_empty_env_still_yields_the_bundled_ontology(self) -> None:
        """docker-compose spells an absent variable as "" — that must not disable it."""
        default = GraphKnowsSettings().ontology_source
        assert GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY="").ontology_source == default
        assert GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY="   ").ontology_source == default

    def test_relations_resolve_to_a_different_bundled_vocabulary(self) -> None:
        """Entity typing and relation extraction do not share a vocabulary.

        schema.org types entities well and supplies relation properties badly
        (945 of them, cosine-selected, 0 relations over 60 real turns), so the
        relation side defaults to the curated profile instead.
        """
        from pathlib import Path

        source = Path(GraphKnowsSettings().relation_ontology_source)
        assert source.parts[-3:] == ("assets", "profile", "personal-profile.json")
        assert source.is_file()
        assert str(source) != GraphKnowsSettings().ontology_source

    def test_empty_relation_env_still_yields_the_bundled_profile(self) -> None:
        default = GraphKnowsSettings().relation_ontology_source
        assert GraphKnowsSettings(GRAPHKNOWS_RELATION_ONTOLOGY="").relation_ontology_source == (
            default
        )
        assert (
            GraphKnowsSettings(GRAPHKNOWS_RELATION_ONTOLOGY="/tmp/x.ttl").relation_ontology_source
            == "/tmp/x.ttl"
        )

    def test_hint_floor_default(self) -> None:
        """Measured over 20 LoCoMo turns against schema.org — see settings.py."""
        assert GraphKnowsSettings().ontology_hint_min_sim == 0.50

    def test_deprecated_alias_still_honoured(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY_FILE="/tmp/legacy.ttl")
        assert s.ontology_source == "/tmp/legacy.ttl"

    def test_new_field_wins_over_alias(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_ONTOLOGY="/tmp/new/",
            GRAPHKNOWS_ONTOLOGY_FILE="/tmp/legacy.ttl",
        )
        assert s.ontology_source == "/tmp/new/"

    def test_ontology_is_mode_independent(self) -> None:
        for mode in (MemoryMode.llm_free, MemoryMode.llm_assisted):
            s = GraphKnowsSettings(GRAPHKNOWS_MODE=mode, GRAPHKNOWS_ONTOLOGY="/tmp/o.ttl")
            assert s.ontology_source == "/tmp/o.ttl"


class TestProductionGuard:
    """The key requirement follows the resolved capability, not the mode label.

    Every case pins ``GRAPHKNOWS_LLM_API_KEY`` explicitly: constructor args beat
    the ambient ``.env``, which otherwise supplies a key and hides the guard.
    """

    def test_llm_free_needs_no_key(self) -> None:
        GraphKnowsSettings(
            GRAPHKNOWS_ENV="production",
            GRAPHKNOWS_ARCADEDB_PASSWORD="secret",
            GRAPHKNOWS_LLM_API_KEY="",
        )  # must not raise

    def test_llm_assisted_with_capabilities_off_needs_no_key(self) -> None:
        GraphKnowsSettings(
            GRAPHKNOWS_ENV="production",
            GRAPHKNOWS_ARCADEDB_PASSWORD="secret",
            GRAPHKNOWS_LLM_API_KEY="",
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_TOPICS=TopicMode.none,
            GRAPHKNOWS_DSPY_RELATIONS=False,
        )  # must not raise
