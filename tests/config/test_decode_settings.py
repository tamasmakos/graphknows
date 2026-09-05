"""The LLM decoder's knobs: defaults, and that the environment moves them.

Pins the nine ``GRAPHKNOWS_DECODE_*`` fields of contracts/public-surface.md §2 —
the section switches that ablate one part of the structured call (FR-016), the
confidence gate whose ``0`` is the fourth ablation arm (FR-014), and the retry
budget with its backoff (FR-020).
"""

from __future__ import annotations

import pytest

from graphknows.settings import GraphKnowsSettings

DEFAULTS: dict[str, bool | int | float] = {
    "decode_entities": True,
    "decode_relations": True,
    "decode_frames": True,
    "decode_confidence_min": 0.5,
    "decode_attempts": 3,
    "decode_backoff_base_s": 1.0,
    "decode_backoff_cap_s": 30.0,
    "decode_honour_retry_after": True,
    "decode_concurrency": 8,
}

OVERRIDES: dict[str, tuple[str, bool | int | float]] = {
    "decode_entities": ("false", False),
    "decode_relations": ("false", False),
    "decode_frames": ("false", False),
    "decode_confidence_min": ("0", 0.0),
    "decode_attempts": ("1", 1),
    "decode_backoff_base_s": ("0.25", 0.25),
    "decode_backoff_cap_s": ("5", 5.0),
    "decode_honour_retry_after": ("false", False),
    "decode_concurrency": ("2", 2),
}


class TestDefaults:
    @pytest.mark.parametrize(("field", "expected"), DEFAULTS.items())
    def test_default(self, field: str, expected: object) -> None:
        assert getattr(GraphKnowsSettings(), field) == expected


class TestEnvironmentOverride:
    @pytest.mark.parametrize(("field", "case"), OVERRIDES.items())
    def test_env_var_moves_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        case: tuple[str, object],
    ) -> None:
        raw, expected = case
        monkeypatch.setenv(f"GRAPHKNOWS_{field.upper()}", raw)
        assert getattr(GraphKnowsSettings(), field) == expected


class TestBounds:
    """Env vars are a trust boundary; a nonsense budget fails at construction."""

    @pytest.mark.parametrize(
        ("field", "raw"),
        [
            ("decode_attempts", "0"),
            ("decode_concurrency", "0"),
            ("decode_confidence_min", "1.5"),
        ],
    )
    def test_rejected(self, monkeypatch: pytest.MonkeyPatch, field: str, raw: str) -> None:
        monkeypatch.setenv(f"GRAPHKNOWS_{field.upper()}", raw)
        with pytest.raises(ValueError):
            GraphKnowsSettings()
