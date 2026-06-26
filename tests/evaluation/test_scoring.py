"""Unit tests for the scoring metrics."""

from __future__ import annotations

from evaluation.datamodels import RetrievedPassage
from evaluation.scoring import (
    answer_tokens,
    gold_max_overlap,
    gold_reachable,
    normalize_answer,
    token_f1,
)


def test_token_f1_exact_match() -> None:
    assert token_f1("Prague", "Prague") == 1.0


def test_normalize_answer_strips_articles_and_punctuation() -> None:
    assert normalize_answer("The Cello!") == "cello"
    assert normalize_answer("yes.") == "yes"
    assert normalize_answer("a paediatric  nurse,") == "paediatric nurse"


def test_token_f1_ignores_articles_and_punctuation() -> None:
    # "the cello" vs "cello" is a perfect match after SQuAD normalization.
    assert token_f1("the cello", "cello") == 1.0
    assert token_f1("yes", "yes.") == 1.0


def test_token_f1_multiset_counts_repeats() -> None:
    # Multiset intersection: a single gold "no" should not be fully covered by
    # a prediction repeating other tokens.
    assert token_f1("paris", "paris paris paris") < 1.0


def test_token_f1_penalises_verbosity() -> None:
    # The gold is one token; a verbose answer that contains it scores < 1.0.
    score = token_f1("Sweden", "my home country, Sweden, is lovely")
    assert 0.0 < score < 1.0


def test_token_f1_empty_or_disjoint_is_zero() -> None:
    assert token_f1("Prague", "") == 0.0
    assert token_f1("", "Prague") == 0.0
    assert token_f1("Prague", "Berlin") == 0.0


def test_answer_tokens_lowercases_and_splits() -> None:
    assert answer_tokens("Prague, 2023!") == {"prague", "2023"}


def test_gold_reachable_requires_all_tokens() -> None:
    passages = [
        RetrievedPassage(text="They flew to Prague in March"),
        RetrievedPassage(text="It was 2023"),
    ]
    assert gold_reachable("Prague 2023", passages) is True
    assert gold_reachable("Vienna 2023", passages) is False


def test_gold_max_overlap_is_best_single_passage() -> None:
    passages = [
        RetrievedPassage(text="Prague only"),
        RetrievedPassage(text="Prague in 2023"),
    ]
    assert gold_max_overlap("Prague 2023", passages) == 1.0
    assert gold_max_overlap("missing entirely", passages) == 0.0
