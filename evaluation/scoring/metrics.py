"""Scoring metrics for LoCoMo generation-mode evaluation.

The headline metric is symmetric token F1 (the SQuAD formula) between the
generated answer and the gold answer — the official LoCoMo paper methodology
(arXiv 2402.17753). The remaining helpers are retrieval diagnostics.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from evaluation.datamodels import RetrievedPassage

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD answer normalization: lowercase, strip punctuation/articles, fix whitespace.

    This is the canonical normalization the LoCoMo paper's F1 scorer inherits from
    SQuAD (arXiv 2402.17753). Without it ``"the cello"`` vs ``"cello"`` — or
    ``"yes."`` vs ``"yes"`` — score as a mismatch.
    """
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token set (used by the retrieval diagnostics)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(gold: str, predicted: str) -> float:
    """SQuAD-style multiset token F1 between a generated answer and the gold answer.

    Uses a multiset (Counter) intersection over SQuAD-normalized tokens — the exact
    LoCoMo paper formula. Penalises both missing gold tokens and added irrelevant
    tokens. Returns 0.0 when either side is empty or they share no tokens.
    """
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(predicted).split()
    if not gold_toks or not pred_toks:
        return 0.0
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
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
