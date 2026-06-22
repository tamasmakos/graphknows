"""Scoring metrics for LoCoMo generation-mode evaluation.

The headline metric is symmetric token F1 (the SQuAD formula) between the
generated answer and the gold answer — the official LoCoMo paper methodology
(arXiv 2402.17753). The remaining helpers are retrieval diagnostics.
"""

from __future__ import annotations

import re

from evaluation.datamodels import RetrievedPassage


def answer_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(gold: str, predicted: str) -> float:
    """Symmetric token F1 between a generated answer and the gold answer.

    Penalises both missing gold tokens and added irrelevant tokens, matching the
    LoCoMo paper. Returns 0.0 when either side is empty or they share no tokens.
    """
    gold_toks = answer_tokens(gold)
    pred_toks = answer_tokens(predicted)
    if not gold_toks or not pred_toks:
        return 0.0
    common = gold_toks & pred_toks
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


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
