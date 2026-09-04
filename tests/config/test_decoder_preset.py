"""The mode picks the decoder, and only the mode picks it.

Pins the preset expansion (FR-003), the ``uses_llm`` predicate that gates the
production-secret check (FR-005), and the retirement of the per-capability
"LLM relations" knob, which under a *replacement* decoder names nothing.
"""

from __future__ import annotations

from graphknows.settings import Decoder, GraphKnowsSettings, MemoryMode, TopicMode


class TestPresetExpansion:
    def test_llm_free_decodes_locally(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert s.decoder is Decoder.local
        assert s.uses_llm is False

    def test_llm_assisted_decodes_with_the_llm(self) -> None:
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted)
        assert s.decoder is Decoder.llm
        assert s.uses_llm is True

    def test_default_is_the_local_decoder(self) -> None:
        assert GraphKnowsSettings().decoder is Decoder.local


class TestDecoderIsReadOnly:
    def test_no_environment_variable_of_its_own(self) -> None:
        """The decoder is resolved, never configured beside the mode."""
        s = GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_free, GRAPHKNOWS_DECODER="llm")
        assert s.decoder is Decoder.local

    def test_the_retired_knob_is_gone(self) -> None:
        assert not hasattr(GraphKnowsSettings(), "dspy_relations")


class TestUsesLlm:
    """``decoder is llm or topic_mode is llm`` — both disjuncts, both ways."""

    def test_llm_topics_alone_reach_the_network(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
            GRAPHKNOWS_TOPICS=TopicMode.llm,
        )
        assert s.decoder is Decoder.local
        assert s.uses_llm is True

    def test_the_llm_decoder_alone_reaches_the_network(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_TOPICS=TopicMode.none,
        )
        assert s.uses_llm is True

    def test_extractive_topics_stay_offline(self) -> None:
        s = GraphKnowsSettings(
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
            GRAPHKNOWS_TOPICS=TopicMode.extractive,
        )
        assert s.uses_llm is False
