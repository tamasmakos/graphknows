"""Lexical scoring metrics and retrieval diagnostics (no LLM calls).

``token_f1``/``locomo_f1`` reproduce the official LoCoMo scorer verbatim from
``task_eval/evaluation.py`` in snap-research/locomo (arXiv 2402.17753): SQuAD
normalization (also stripping "and"), **Porter stemming**, multiset token F1;
multi-hop answers are split on commas and scored per gold sub-answer; temporal
gold is truncated at the first ``;``. Matching it exactly keeps our numbers
directly comparable to the paper. The remaining helpers are retrieval
diagnostics shared by all benchmarks.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from evaluation.common.datamodels import RetrievedPassage

# LoCoMo removes a/an/the AND "and" (unlike plain SQuAD).
_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCT = str.maketrans("", "", string.punctuation)

_stemmer: Any = None


def _stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer

        _stemmer = PorterStemmer()
    return _stemmer.stem(word)


def normalize_answer(text: str) -> str:
    """LoCoMo answer normalization: drop commas, lowercase, strip a/an/the/and,
    strip punctuation, fix whitespace (verbatim from the official scorer)."""
    text = text.replace(",", "").lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())


def answer_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token set (used by the retrieval diagnostics)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(gold: str, predicted: str) -> float:
    """Official LoCoMo token F1: multiset F1 over Porter-stemmed, normalized tokens.

    Stemming means "running"/"run" and "researched"/"research" match, as in the
    paper. Penalises both missing gold tokens and added irrelevant ones.
    """
    gold_toks = [_stem(w) for w in normalize_answer(gold).split()]
    pred_toks = [_stem(w) for w in normalize_answer(predicted).split()]
    if not gold_toks or not pred_toks:
        return 0.0
    num_same = sum((Counter(gold_toks) & Counter(pred_toks)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


def locomo_f1(gold: str, predicted: str, category: str) -> float:
    """Category-aware official LoCoMo F1 (see ``eval_question_answering``).

    * multi-hop (``knowledge_synthesis``): split both sides on commas and average,
      per gold sub-answer, the best matching predicted sub-answer.
    * temporal: score against the gold truncated at the first ``;``.
    * single-hop / open-ended: plain stemmed token F1.
    """
    if category == "temporal":
        gold = gold.split(";")[0].strip()
    if category == "knowledge_synthesis":
        preds = [p.strip() for p in predicted.split(",")]
        golds = [g.strip() for g in gold.split(",")]
        return sum(max(token_f1(p, g) for p in preds) for g in golds) / len(golds)
    return token_f1(gold, predicted)


def gold_reachable(gold: str, passages: list[RetrievedPassage]) -> bool:
    """Whether every gold token appears somewhere across the passages.

    Lenient — gold tokens may be scattered, so this overstates how many answers
    are genuinely present. Used to separate retrieval misses from generation
    misses, not as a correctness signal.
    """
    gold_toks = answer_tokens(gold)
    if not gold_toks:
        return False
    haystack = answer_tokens("\n".join(p.text for p in passages))
    return gold_toks <= haystack


def evidence_recall(evidence: list[str], texts: list[str]) -> float:
    """Fraction of gold EVIDENCE turns present verbatim in the recalled *texts*.

    The honest retrieval metric, where ``gold_reachable`` is a proxy: it scores
    whether we retrieved the turns that actually answer the question, not
    whether the answer's words happen to be scattered across the context. So it
    is per-hop (a 4-evidence question can score 0.25) and cannot be inflated by
    enlarging the haystack — irrelevant passages add nothing to the numerator.

    Exact substring on whitespace-normalised text: verified 97/97 on conv-26,
    since chunk text preserves the utterance verbatim. Callers skip cases with
    no evidence rather than scoring them 0.

    Takes plain strings, not ``RetrievedPassage``: it needs nothing but the
    text, and demanding the harness's passage type is what made an out-of-
    harness caller (the LangGraph runner, holding ``Hit``s) reimplement it.
    """
    if not evidence:
        return 0.0
    haystack = " ".join(" ".join(t.split()) for t in texts)
    return sum(1 for turn in evidence if turn in haystack) / len(evidence)


def gold_max_overlap(gold: str, passages: list[RetrievedPassage]) -> float:
    """Best single-passage gold-token recall across the passages (0.0-1.0)."""
    gold_toks = answer_tokens(gold)
    if not gold_toks:
        return 0.0
    best = 0.0
    for passage in passages:
        overlap = len(gold_toks & answer_tokens(passage.text)) / len(gold_toks)
        best = max(best, overlap)
    return best


def gold_context_coverage(gold: str, passages: list[RetrievedPassage]) -> float:
    """Fraction of the gold's content tokens present in the generator's passages.

    High coverage + wrong answer = generation failure; low coverage = retrieval
    miss. Splitting the two per-case makes failure attribution readable straight
    from the results JSON.
    """
    word_re = re.compile(r"[a-z0-9']+")
    gold_toks = {w for w in word_re.findall((gold or "").lower()) if len(w) > 2}
    if not gold_toks:
        return 0.0
    ctx = " ".join(p.text for p in passages).lower()
    ctx_toks = set(word_re.findall(ctx))
    return len(gold_toks & ctx_toks) / len(gold_toks)


def kendall_tau_b(predicted: list[int], reference: list[int]) -> float:
    """Kendall tau-b rank correlation between two orderings, with tie handling.

    Inputs are parallel lists of rank positions; returns a value in [-1, 1]
    (0.0 when either list has fewer than 2 items or all pairs are ties).
    """
    n = min(len(predicted), len(reference))
    if n < 2:
        return 0.0
    concordant = discordant = ties_pred = ties_ref = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = predicted[i] - predicted[j]
            dr = reference[i] - reference[j]
            if dp == 0 and dr == 0:
                continue
            if dp == 0:
                ties_pred += 1
            elif dr == 0:
                ties_ref += 1
            elif (dp > 0) == (dr > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_pred) * (concordant + discordant + ties_ref)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom
