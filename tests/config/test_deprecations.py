"""The retired per-capability relation knob deprecates loudly and does nothing.

FR-004 / Scenario 4.3: ``GRAPHKNOWS_DSPY_RELATIONS`` is neither silently
honoured nor silently dropped — setting it warns, and the value is discarded.
"""

from __future__ import annotations

import warnings

import pytest

from graphknows.settings import Decoder, GraphKnowsSettings, MemoryMode, TopicMode


class TestRetiredRelationsKnob:
    @pytest.mark.parametrize("value", [True, False, "true", "false"])
    def test_setting_it_warns_and_names_the_replacement(self, value: bool | str) -> None:
        with pytest.warns(DeprecationWarning, match=r"Memory\(mode=\.\.\.\)") as caught:
            GraphKnowsSettings(GRAPHKNOWS_DSPY_RELATIONS=value)
        assert any("GRAPHKNOWS_DSPY_RELATIONS" in str(w.message) for w in caught)

    @pytest.mark.parametrize("value", [True, False])
    def test_the_value_is_dropped_not_stored(self, value: bool) -> None:
        """Nothing downstream can honour a field that never holds a value."""
        with pytest.warns(DeprecationWarning):
            s = GraphKnowsSettings(GRAPHKNOWS_DSPY_RELATIONS=value)
        assert s.enable_dspy_relations is None

    @pytest.mark.parametrize("mode", list(MemoryMode))
    def test_it_moves_no_resolved_capability(self, mode: MemoryMode) -> None:
        with pytest.warns(DeprecationWarning):
            forced = GraphKnowsSettings(GRAPHKNOWS_MODE=mode, GRAPHKNOWS_DSPY_RELATIONS=True)
        preset = GraphKnowsSettings(GRAPHKNOWS_MODE=mode)
        assert forced.decoder is preset.decoder
        assert forced.topic_mode is preset.topic_mode
        assert forced.uses_llm is preset.uses_llm

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_env_var_is_not_a_deprecation(self, value: str) -> None:
        """docker-compose spells an absent variable as ``VAR=`` — that is unset."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            s = GraphKnowsSettings(GRAPHKNOWS_DSPY_RELATIONS=value)
        assert s.enable_dspy_relations is None
        assert s.decoder is Decoder.local
        assert s.topic_mode is TopicMode.none
