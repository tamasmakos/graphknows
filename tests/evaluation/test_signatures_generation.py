"""Structural guards for the LoCoMo answer-generation signature.

These assert the prompt-facing rules/demos are present and dataset-agnostic, with
no network/LLM. The clock-time guard protects the fix for temporal
over-specification (e.g. "2:35 pm on 16 March 2023" must answer "16 March 2023").
"""
from __future__ import annotations

import re

from evaluation.datamodels import RetrievedPassage
from evaluation.generation.signatures import LoCoMoAnswerGeneration, build_context, demos


def test_clock_time_rule_present():
    text = (LoCoMoAnswerGeneration.__doc__ or "").lower()
    assert "clock time" in text
    assert "granularity" in text


def test_a_demo_strips_clock_time():
    """At least one demo has a clock time in the context but not in the answer."""
    clocked = [
        d for d in demos()
        if re.search(r"\d{1,2}:\d{2}\s*(am|pm)", d.retrieved_context, re.I)
    ]
    assert clocked, "expected a demo whose context carries a clock time"
    for d in clocked:
        assert not re.search(r"\d{1,2}:\d{2}", d.answer), (
            f"demo answer must not carry a clock time: {d.answer!r}"
        )


def test_demos_are_dataset_agnostic():
    """Demos teach behaviour with invented people — never leak benchmark content."""
    blob = " ".join(f"{d.question} {d.retrieved_context} {d.answer}" for d in demos()).lower()
    for banned in ("caroline", "melanie", "gina", "nate", "joanna", "locomo"):
        assert banned not in blob


def test_build_context_dedupes_and_numbers():
    ctx = build_context([
        RetrievedPassage(text="alpha", score=1.0),
        RetrievedPassage(text="alpha", score=0.9),  # duplicate dropped
        RetrievedPassage(text="beta", score=0.8),
    ])
    assert ctx == "[1] alpha\n\n[2] beta"
