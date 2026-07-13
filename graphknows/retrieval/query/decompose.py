"""Query decomposition helper for agentic retrieval.

Splits a multi-hop or temporally complex question into a list of atomic
sub-questions, each answerable by a single deterministic retrieval hop.

The decomposer is intentionally lightweight:
- It uses DSPy with the same OpenRouter LM already configured in agentic.py.
- It has a deterministic single-element fallback: when the LM is unavailable
  or raises any exception, ``decompose()`` returns ``[query]`` (the original
  question as-is), so the caller always receives a non-empty list and can
  proceed to retrieval without any special-casing.
- The module is importable without a live API (the DSPy import itself is
  guarded; no network call is made at import time).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Maximum number of sub-questions the decomposer may return.
# Keeps the ReAct hop budget reasonable.
MAX_SUBQUESTIONS: int = 4


try:
    import dspy as _dspy
except ImportError:  # pragma: no cover - optional deployment dependency
    _dspy = None  # type: ignore[assignment]


def _require_dspy() -> Any:
    if _dspy is None:
        raise ImportError("dspy is required for query decomposition")
    return _dspy


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------


def _build_signature(dspy_mod: Any) -> type:
    """Construct and return the DSPy Signature class using the live dspy module."""

    class DecomposeSignature(dspy_mod.Signature):  # type: ignore[misc, name-defined]
        """Split a complex question into atomic sub-questions for retrieval.

        Given a user question that may involve multiple entities, time periods,
        or reasoning hops, produce a numbered list of simple, self-contained
        sub-questions. Each sub-question should be answerable by a single
        retrieval call. If the question is already simple (single-hop), return
        it unchanged as the only item.

        Output format: one sub-question per line, each prefixed with a number
        and a period (e.g. "1. What is ...\\n2. When did ...").
        Return at most four sub-questions.
        """

        question: str = dspy_mod.InputField(
            desc="The original user question, potentially multi-hop or temporal."
        )
        sub_questions: str = dspy_mod.OutputField(
            desc=(
                "Newline-separated list of atomic sub-questions, each prefixed "
                "with '<n>. ' where n is the question number."
            )
        )

    return DecomposeSignature


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose(
    query: str,
    *,
    lm: Any = None,
) -> list[str]:
    """Decompose *query* into atomic sub-questions.

    Args:
        query: The original user question.
        lm: Optional pre-configured DSPy LM instance.  When ``None`` the
            function attempts to use whatever LM is active in the current
            DSPy context.  When no LM is available or the call fails, the
            function falls back to ``[query]``.

    Returns:
        A non-empty list of sub-question strings.  The list always contains
        at least ``[query]`` (the deterministic fallback).
    """
    if not query or not query.strip():
        return [query]

    try:
        dspy_mod = _require_dspy()
        sig = _build_signature(dspy_mod)
        predictor = dspy_mod.Predict(sig)

        ctx_kwargs: dict[str, Any] = {}
        if lm is not None:
            ctx_kwargs["lm"] = lm

        if ctx_kwargs:
            with dspy_mod.settings.context(**ctx_kwargs):
                result = predictor(question=query)
        else:
            result = predictor(question=query)

        raw: str = (result.sub_questions or "").strip()
        if not raw:
            return [query]

        sub_qs = _parse_sub_questions(raw, query)
        return sub_qs or [query]

    except Exception as exc:  # noqa: BLE001
        log.debug("decompose fell back to [query]: %s", exc)
        return [query]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_sub_questions(raw: str, fallback: str) -> list[str]:
    """Parse newline-separated numbered sub-questions from *raw*.

    Accepts:
        "1. What is X?\\n2. When did Y happen?"
        "1) What is X?\\n2) When did Y happen?"
        Plain lines without numbering are included as-is.

    Returns at most ``MAX_SUBQUESTIONS`` items.  Returns ``[]`` when no
    parseable content is found (caller should fall back to ``[fallback]``).
    """
    import re

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    sub_qs: list[str] = []
    _number_prefix = re.compile(r"^\d+[.)]\s*")

    for line in lines:
        stripped = _number_prefix.sub("", line).strip()
        if stripped:
            sub_qs.append(stripped)
        if len(sub_qs) >= MAX_SUBQUESTIONS:
            break

    return sub_qs
