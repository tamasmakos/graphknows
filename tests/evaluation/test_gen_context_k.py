"""gen_context_k depth knob: env override, probe_k auto-coverage, CLI flag."""

from __future__ import annotations

import pytest

from evaluation.common.config import BaseEvalConfig
from evaluation.locomo.cli import build_parser
from evaluation.locomo.config import LocomoConfig


def test_default_unchanged() -> None:
    """Defaults are the shipped baseline: gen_context_k=25, probe_k=50.

    gen_context_k went 25 -> 50 when gold evidence was landing at ranks 26-36
    and getting cut, then back to 25 once cosine ordering of the fused pool
    stopped burying it: evidence_recall@25 is 0.953 vs @50's 0.993, while
    accuracy on conv-26 goes 0.842 (ctx=50) -> 0.882 (ctx=25) -> 0.829 (ctx=10).
    The window was never the problem; the ranking was.
    """
    cfg = BaseEvalConfig()
    assert cfg.gen_context_k == 25
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
