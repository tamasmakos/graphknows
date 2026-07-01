"""Tests for mode-specific gen_context_k selection (select_gen_context_k helper)."""

from evaluation.config import EvalConfig
from evaluation.pipeline import select_gen_context_k


def test_config_defaults():
    """EvalConfig default field values match the documented split."""
    cfg = EvalConfig()
    assert cfg.gen_context_k == 25, "llm_assisted / default should be 25"
    assert cfg.gen_context_k_free == 25, "llm_free efficiency default (equal accuracy to 50, -18% tokens)"


def test_llm_free_mode_uses_free_value():
    cfg = EvalConfig()
    assert select_gen_context_k(cfg, "llm_free") == cfg.gen_context_k_free


def test_llm_assisted_mode_uses_default():
    cfg = EvalConfig()
    assert select_gen_context_k(cfg, "llm_assisted") == cfg.gen_context_k


def test_unset_mode_uses_default():
    cfg = EvalConfig()
    assert select_gen_context_k(cfg, None) == cfg.gen_context_k


def test_unknown_mode_uses_default():
    """Any unrecognised mode falls back to the conservative default."""
    cfg = EvalConfig()
    assert select_gen_context_k(cfg, "some_future_mode") == cfg.gen_context_k


def test_custom_config_values_respected():
    """Helper honours overridden field values, not just the hardcoded defaults."""
    cfg = EvalConfig(gen_context_k=10, gen_context_k_free=20)
    assert select_gen_context_k(cfg, "llm_free") == 20
    assert select_gen_context_k(cfg, "llm_assisted") == 10
