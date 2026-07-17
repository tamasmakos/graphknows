"""gen_context_k depth knob: env override, probe_k auto-coverage, CLI flag."""

from __future__ import annotations

import pytest

from evaluation.common.config import BaseEvalConfig
from evaluation.locomo.cli import build_parser
from evaluation.locomo.config import LocomoConfig


def test_default_unchanged() -> None:
    """Defaults are the shipped baseline: gen_context_k=50, probe_k=50.

    gen_context_k was raised 25 -> 50 after evidence-rank attribution on conv-26
    showed 29 of 81 failures had their gold evidence retrieved at ranks 26-36 —
    fetched by the probe, then cut by the 25-passage window.
    """
    cfg = BaseEvalConfig()
    assert cfg.gen_context_k == 50
    assert cfg.probe_k == 50


def test_probe_k_auto_raises_to_cover_gen_context_k() -> None:
    """probe_k must cover gen_context_k (context = probe[:gen_context_k])."""
    cfg = BaseEvalConfig(gen_context_k=100)
    assert cfg.probe_k >= 100
    cfg50 = BaseEvalConfig(gen_context_k=50)
    assert cfg50.probe_k >= 50


def test_probe_k_not_lowered_when_already_larger() -> None:
    cfg = BaseEvalConfig(gen_context_k=25, probe_k=80)
    assert cfg.probe_k == 80


def test_env_override_sets_gen_context_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVAL_GEN_CONTEXT_K drives gen_context_k, and probe_k follows."""
    monkeypatch.setenv("EVAL_GEN_CONTEXT_K", "100")
    cfg = LocomoConfig()
    assert cfg.gen_context_k == 100
    assert cfg.probe_k >= 100


def test_cli_flag_present_and_parsed() -> None:
    parser = build_parser()
    assert parser.parse_args(["--gen-context-k", "50"]).gen_context_k == 50
    assert parser.parse_args([]).gen_context_k is None
