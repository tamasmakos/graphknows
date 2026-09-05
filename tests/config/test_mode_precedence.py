"""``Memory(mode=...)`` — the one argument that swaps the decoder.

Pins the precedence the argument shares with ``namespace`` (FR-001), that
nothing implicit sets the mode (FR-002, Scenarios 1.3 and 1.4), and that a
keyless LLM decoder fails at construction rather than at the first chunk
(FR-006). See ``contracts/public-surface.md`` §1.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.exceptions import ConfigurationError
from graphknows.memory import Memory
from graphknows.settings import Decoder, GraphKnowsSettings, MemoryMode


def _settings(**kwargs: Any) -> GraphKnowsSettings:
    """Settings read from nothing but these arguments and the patched env."""
    return GraphKnowsSettings(_env_file=None, **kwargs)


@pytest.fixture(autouse=True)
def _no_ambient_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own key or mode in the environment must not steer these."""
    for var in ("GRAPHKNOWS_MODE", "GRAPHKNOWS_LLM_API_KEY", "GRAPHKNOWS_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


class TestPrecedence:
    def test_the_argument_wins_over_the_settings_value(self) -> None:
        """Scenario 1.3: settings say llm_free, the argument says llm_assisted."""
        settings = _settings(GRAPHKNOWS_MODE=MemoryMode.llm_free, GRAPHKNOWS_LLM_API_KEY="k")
        mem = Memory(settings, mode="llm_assisted")
        assert mem.settings.mode is MemoryMode.llm_assisted
        assert mem.settings.decoder is Decoder.llm

    def test_a_memory_mode_member_is_accepted_too(self) -> None:
        mem = Memory(_settings(GRAPHKNOWS_LLM_API_KEY="k"), mode=MemoryMode.llm_assisted)
        assert mem.settings.mode is MemoryMode.llm_assisted

    def test_no_argument_leaves_the_settings_value(self) -> None:
        settings = _settings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted, GRAPHKNOWS_LLM_API_KEY="k")
        assert Memory(settings).settings.mode is MemoryMode.llm_assisted

    def test_the_callers_settings_object_is_not_mutated(self) -> None:
        settings = _settings(GRAPHKNOWS_MODE=MemoryMode.llm_free, GRAPHKNOWS_LLM_API_KEY="k")
        mem = Memory(settings, mode="llm_assisted")
        assert settings.mode is MemoryMode.llm_free
        assert mem.settings is not settings

    def test_namespace_still_resolves_beside_the_mode(self) -> None:
        settings = _settings(GRAPHKNOWS_NAMESPACE="from_settings", GRAPHKNOWS_LLM_API_KEY="k")
        assert Memory(settings, namespace="explicit", mode="llm_assisted").namespace == "explicit"
        assert Memory(settings).namespace == "from_settings"


class TestNothingImplicitSetsTheMode:
    def test_a_key_and_model_in_the_environment_leave_it_llm_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario 1.4 / FR-002."""
        monkeypatch.setenv("GRAPHKNOWS_LLM_API_KEY", "sk-present")
        monkeypatch.setenv("GRAPHKNOWS_LLM_MODEL", "openrouter/some/model")
        monkeypatch.setenv("GRAPHKNOWS_LLM_API_BASE", "https://example.invalid/v1")
        mem = Memory(_settings())
        assert mem.settings.mode is MemoryMode.llm_free
        assert mem.settings.decoder is Decoder.local


class TestKeyResolvesAtConstruction:
    def test_the_assisted_mode_without_a_key_raises_naming_the_variable(self) -> None:
        """FR-006: fail here, not at the first chunk of a long ingest."""
        with pytest.raises(ConfigurationError, match="GRAPHKNOWS_LLM_API_KEY"):
            Memory(_settings(), mode="llm_assisted")

    def test_the_settings_value_is_checked_too(self) -> None:
        with pytest.raises(ConfigurationError, match="GRAPHKNOWS_LLM_API_KEY"):
            Memory(_settings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted))

    def test_the_local_decoder_needs_no_key(self) -> None:
        assert Memory(_settings()).settings.decoder is Decoder.local

    def test_the_production_check_is_re_run_after_the_mode_swap(self) -> None:
        """``model_copy`` skips validators, so the check runs by hand."""
        settings = _settings(
            GRAPHKNOWS_ENV="production",
            GRAPHKNOWS_ARCADEDB_PASSWORD="not-the-default",
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
        )
        with pytest.raises(ConfigurationError, match="production"):
            Memory(settings, mode="llm_assisted")
