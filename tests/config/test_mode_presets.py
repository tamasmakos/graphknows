"""MemoryMode is a preset over orthogonal knobs, not a global fork.

Pins the expansion table and the override precedence, so the "resolved
capability" contract every consumer relies on cannot drift silently.
"""

from __future__ import annotations

import pytest

from graphknows.exceptions import ConfigurationError
from graphknows.settings import (
    Decoder,
    GraphKnowsSettings,
    MemoryMode,
    TopicMode,
)


class TestPresetExpansion:
    def test_llm_free_preset(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert s.topic_mode is TopicMode.none
        assert s.decoder is Decoder.local
        assert s.uses_llm is False

    def test_llm_assisted_preset(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted)
        assert s.topic_mode is TopicMode.llm
        assert s.decoder is Decoder.llm
        assert s.uses_llm is True

    def test_default_matches_llm_free(self) -> None:
        """A bare construction must behave exactly like today's default."""
        s = GraphKnowsSettings()
        assert s.mode is MemoryMode.llm_free
        assert s.topic_mode is TopicMode.none
        assert s.decoder is Decoder.local


class TestKnobOverrides:
    def test_topics_opt_in_under_llm_free(self) -> None:
        """The headline case: deterministic topics without leaving llm_free."""
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
            GRAPHKNOWS_TOPICS=TopicMode.extractive,
        )
        assert s.topic_mode is TopicMode.extractive
        assert s.decoder is Decoder.local  # untouched by the override
        assert s.uses_llm is False  # extractive topics make no network call

    def test_retired_relations_knob_does_not_move_the_decoder(self) -> None:
        """Under a replacement decoder the per-capability knob names nothing."""
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_DSPY_RELATIONS=False,
        )
        assert s.decoder is Decoder.llm
        assert s.topic_mode is TopicMode.llm  # other knobs keep the preset

    def test_llm_topics_alone_reach_the_network(self) -> None:
        """``uses_llm`` follows either resolved capability, not the mode label."""
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
            GRAPHKNOWS_TOPICS=TopicMode.llm,
        )
        assert s.decoder is Decoder.local
        assert s.uses_llm is True


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
        assert s.decoder is Decoder.local


class TestOntologySource:
    def test_unset_resolves_to_the_bundled_cco_digest(self) -> None:
        """Ontology injection is always on: no configuration still yields one."""
        from pathlib import Path

        source = Path(GraphKnowsSettings().ontology_source)
        assert source.parts[-3:] == ("assets", "cco", "cco.json")
        assert source.is_file()

    def test_empty_env_still_yields_the_bundled_ontology(self) -> None:
        """docker-compose spells an absent variable as "" — that must not disable it."""
        default = GraphKnowsSettings().ontology_source
        assert GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY="").ontology_source == default
        assert GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY="   ").ontology_source == default

    def test_relations_resolve_to_the_personal_profile_by_default(self) -> None:
        """Entity typing and relation vocabulary are two separate knobs again.

        The conv-30 personal-vs-cco A/B (evaluation/audit/baseline/
        relation-vocabulary-ab.md) found CCO's object properties leave 13.5% of
        REL edges as unanchored ``lemma:`` predicates versus the personal
        profile's 2.5%, so #137 makes the personal profile the relation
        default while CCO keeps typing entities via ``ontology_source``.
        """
        from pathlib import Path

        source = Path(GraphKnowsSettings().relation_ontology_source)
        assert source.parts[-3:] == ("assets", "personal", "personal-profile.json")
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
        """Measured over 20 LoCoMo turns against the CCO digest — see settings.py."""
        assert GraphKnowsSettings().ontology_hint_min_sim == 0.46

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

    def test_llm_assisted_always_needs_a_key(self) -> None:
        """The decoder is the mode's whole choice: no knob turns its LLM off."""
        with pytest.raises(ConfigurationError):
            GraphKnowsSettings(
                GRAPHKNOWS_ENV="production",
                GRAPHKNOWS_ARCADEDB_PASSWORD="secret",
                GRAPHKNOWS_LLM_API_KEY="",
                GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
                GRAPHKNOWS_TOPICS=TopicMode.none,
                GRAPHKNOWS_DSPY_RELATIONS=False,
            )
