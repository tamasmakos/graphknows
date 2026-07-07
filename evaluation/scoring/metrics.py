"""Scoring metrics for LoCoMo generation-mode evaluation.

The headline metric is the OFFICIAL LoCoMo token F1 — reproduced verbatim from
``task_eval/evaluation.py`` in snap-research/locomo (arXiv 2402.17753): SQuAD
normalization (also stripping "and"), **Porter stemming**, multiset token F1;
multi-hop answers are split on commas and scored per gold sub-answer; temporal
gold is truncated at the first ``;``. Matching it exactly makes our numbers
directly comparable to the paper. The remaining helpers are retrieval diagnostics.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from evaluation.datamodels import RetrievedPassage

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
