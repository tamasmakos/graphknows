"""``build_decoder`` — the one place the memory mode forks (FR-008, research R1).

Three claims: the fork picks the right arm, the local arm stays a process-wide
singleton, and no consumer forks a second time.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities import build_decoder, reset_entity_extractor
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.settings import GraphKnowsSettings, MemoryMode

_LOCAL_ARM = "graphknows.ingestion.extraction.entities.extractor.GLiNER2EntityExtractor"


class _StubExtractor:
    """Stands in for GLiNER2, which loads ~2.5 GB of models."""


@pytest.fixture
def local_arm(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_StubExtractor]]:
    """Stub the local extractor out, and drop the process-wide cache around the test."""
    monkeypatch.setattr(_LOCAL_ARM, _StubExtractor)
    reset_entity_extractor()
    yield _StubExtractor
    reset_entity_extractor()


def _settings(mode: MemoryMode) -> GraphKnowsSettings:
    return GraphKnowsSettings(GRAPHKNOWS_MODE=mode)


# --- The fork ----------------------------------------------------------------


def test_llm_mode_decodes_with_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-008: the LLM arm is chosen, and GLiNER2 is not loaded to get there."""

    def _explode() -> Any:
        raise AssertionError("the local extractor was constructed in llm_assisted")

    monkeypatch.setattr(_LOCAL_ARM, _explode)

    assert isinstance(build_decoder(_settings(MemoryMode.llm_assisted)), LLMDecoder)


def test_local_mode_decodes_with_gliner(local_arm: type[_StubExtractor]) -> None:
    assert isinstance(build_decoder(_settings(MemoryMode.llm_free)), local_arm)


# --- The caching -------------------------------------------------------------


def test_the_local_arm_is_one_per_process(local_arm: type[_StubExtractor]) -> None:
    settings = _settings(MemoryMode.llm_free)

    assert build_decoder(settings) is build_decoder(settings)


def test_reset_releases_the_local_arm(local_arm: type[_StubExtractor]) -> None:
    settings = _settings(MemoryMode.llm_free)
    first = build_decoder(settings)
    reset_entity_extractor()

    assert build_decoder(settings) is not first


def test_the_llm_arm_is_per_caller() -> None:
    """It costs one provider handle to build and carries per-run counters."""
    settings = _settings(MemoryMode.llm_assisted)

    assert build_decoder(settings) is not build_decoder(settings)


# --- The only fork -----------------------------------------------------------


def test_the_mode_is_read_exactly_once() -> None:
    assert inspect.getsource(build_decoder).count("settings.decoder") == 1


def test_the_composition_root_does_not_fork() -> None:
    """``build_stm_service`` composes; it must not know the modes apart."""
    root = Path(inspect.getfile(build_decoder)).parents[3] / "ingestion" / "__init__.py"

    assert "Decoder" not in root.read_text(encoding="utf-8")
