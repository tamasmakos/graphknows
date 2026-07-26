"""DSPy judge: (question, gold, generated) → a 0..1 correctness score.

Only the *judge* uses DSPy — answer generation stays a plain chat completion
(:mod:`evaluation.common.llm`). A single ``dspy.Predict`` over the signature
below is fired with the same model we use everywhere (``LLM_MODEL`` / the
optional ``JUDGE_MODEL`` override) through OpenRouter, and called concurrently
via DSPy's native async path (``acall``).
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

from evaluation.common.lexical import token_f1
from graphknows.llm import get_openrouter_api_key

log = logging.getLogger(__name__)


class JudgeSignature(dspy.Signature):
    """Grade whether the generated answer correctly answers the question, using the gold answer as ground truth.

    Be lenient about surface form: paraphrases, supersets that still contain the
    gold fact, and differently formatted dates that convey the same day are
    correct. Score 1.0 when the generated answer conveys the gold answer, 0.0 when
    it is wrong, missing, or a refusal (unless the gold itself is a refusal), and a
    value in between for a partially correct answer (e.g. some items of a list).
    """

    question: str = dspy.InputField(desc="the question asked")
    gold: str = dspy.InputField(desc="the reference (golden) answer")
    generated: str = dspy.InputField(desc="the answer to evaluate")
    reasoning: str = dspy.OutputField(desc="one sentence justifying the score")
    score: float = dspy.OutputField(desc="correctness from 0.0 (wrong) to 1.0 (fully correct)")


class VerdictSignature(dspy.Signature):
    """Deliver a final correctness verdict, using two quantitative signals as evidence.

    You are given two scores for the generated answer: ``semantic_f1`` (an LM's
    estimate of how much of the gold answer's *meaning* the generated answer
    conveys) and ``token_f1`` (lexical token overlap between the two strings).
    Weigh both as evidence, but rule on whether the generated answer is ACTUALLY
    correct given the gold answer — the signals inform your judgement, they do not
    decide it. A high ``token_f1`` with the wrong meaning is still wrong (e.g. the
    same words rearranged into a false claim); a low ``token_f1`` that is a correct
    paraphrase, a superset containing the gold fact, or a differently formatted but
    equivalent date is still right. Score 1.0 when the answer conveys the gold, 0.0
    when it is wrong, missing, or a refusal (unless the gold is itself a refusal),
    and a value in between for a partially correct answer (e.g. some items of a list).
    """

    question: str = dspy.InputField(desc="the question asked")
    gold: str = dspy.InputField(desc="the reference (golden) answer")
    generated: str = dspy.InputField(desc="the answer to evaluate")
    semantic_f1: float = dspy.InputField(
        desc="stage-1 LM semantic-overlap score, 0.0-1.0 (evidence, not a verdict)"
    )
    token_f1: float = dspy.InputField(
        desc="lexical token-overlap F1, 0.0-1.0 (evidence, not a verdict)"
    )
    reasoning: str = dspy.OutputField(desc="one sentence justifying the final grade")
    score: float = dspy.OutputField(
        desc="final correctness from 0.0 (wrong) to 1.0 (fully correct)"
    )


class RubricSignature(dspy.Signature):
    """Grade whether the generated answer satisfies a rubric, when there is no single gold answer.

    Some questions (e.g. BEAM instruction/preference following) have no reference
    string — correctness is defined by a rubric of criteria the answer must meet.
    Score 1.0 when the answer satisfies all criteria, 0.0 when it violates them or
    ignores the instruction, and a value in between for partial satisfaction.
    """

    question: str = dspy.InputField(desc="the question or instruction asked")
    rubric: str = dspy.InputField(desc="the grading criteria the answer must satisfy")
    generated: str = dspy.InputField(desc="the answer to evaluate")
    reasoning: str = dspy.OutputField(desc="one sentence justifying the score")
    score: float = dspy.OutputField(
        desc="rubric satisfaction from 0.0 (fails) to 1.0 (fully meets)"
    )


class DSPyJudge:
    """One-LM DSPy judge bound to the configured OpenRouter model.

    Exposes graders over the same LM: :meth:`judge` (stage-1 gold-string
    SemanticF1), :meth:`judge_two_stage` (stage-1 SemanticF1 + lexical token_f1
    fed as evidence into a stage-2 correctness verdict), and :meth:`judge_rubric`
    (criteria satisfaction, for questions with no single reference answer).
    """

    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        if not model:
            raise RuntimeError("No judge model configured. Set LLM_MODEL/JUDGE_MODEL in .env.")
        key = get_openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
        # litellm (DSPy's backend) needs the openrouter/ route prefix.
        lm_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        # Build through graphknows.llm.build_lm rather than dspy.LM directly: it
        # is the single place the timeout and the OpenRouter api_base are pinned.
        # Constructing the LM here instead meant the judge had neither — no
        # timeout (a run wedged 2h49m on one call) and no pinned base (litellm
        # can strip the openrouter/ prefix and mis-route the nested provider).
        from graphknows.llm import build_lm

        self._lm = build_lm(lm_model, api_key=key, temperature=temperature, max_tokens=max_tokens)
        dspy.configure(lm=self._lm)
        self._predict = dspy.Predict(JudgeSignature)
        self._verdict_predict = dspy.Predict(VerdictSignature)
        self._rubric_predict = dspy.Predict(RubricSignature)

    async def judge(self, question: str, gold: str, generated: str) -> tuple[float, str]:
        """Return ``(score_in_0_1, reasoning)`` for one gold-string triple."""
        pred = await self._predict.acall(question=question, gold=gold, generated=generated)
        return _clamp01(getattr(pred, "score", 0.0)), str(getattr(pred, "reasoning", "")).strip()

    async def judge_two_stage(
        self, question: str, gold: str, generated: str
    ) -> tuple[float, dict[str, Any]]:
        """Two-stage grade: return ``(stage2_score, metrics)``.

        Stage 1 is the unchanged SemanticF1 :meth:`judge` (LM semantic overlap).
        The lexical ``token_f1`` is computed cheaply. Both are then fed as evidence
        into a stage-2 :class:`VerdictSignature` prediction that rules on actual
        correctness; its score is the headline. ``metrics`` exposes both
        ``semantic_f1`` (stage 1) and ``judge_score`` (stage 2 final) plus
        ``token_f1`` so their divergence is inspectable per case.
        """
        semantic_f1, _ = await self.judge(question, gold, generated)
        lexical_f1 = token_f1(gold, generated)
        pred = await self._verdict_predict.acall(
            question=question,
            gold=gold,
            generated=generated,
            semantic_f1=semantic_f1,
            token_f1=lexical_f1,
        )
        score = _clamp01(getattr(pred, "score", 0.0))
        reasoning = str(getattr(pred, "reasoning", "")).strip()
        return score, {
            "semantic_f1": semantic_f1,
            "token_f1": lexical_f1,
            "judge_score": score,
            "judge_reasoning": reasoning,
        }

    async def judge_rubric(self, question: str, rubric: str, generated: str) -> tuple[float, str]:
        """Return ``(score_in_0_1, reasoning)`` grading an answer against a rubric."""
        pred = await self._rubric_predict.acall(
            question=question, rubric=rubric, generated=generated
        )
        return _clamp01(getattr(pred, "score", 0.0)), str(getattr(pred, "reasoning", "")).strip()

    async def aclose(self) -> None:
        """No persistent connection to close; kept for ChatLLM API parity."""
        return None


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
