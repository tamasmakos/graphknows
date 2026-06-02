"""Unit tests for HotpotQA evaluation helpers."""

from __future__ import annotations

from evaluation.benchmarks.hotpotqa import score_hotpotqa_support


def test_score_hotpotqa_support_matches_titles_and_answer() -> None:
    metrics = score_hotpotqa_support(
        [{"text": "Title: The Odyssey\nHomer wrote the Odyssey. Ancient Greece appears here."}],
        supporting_titles=["The Odyssey", "Ancient Greece"],
        answer="Homer",
    )

    assert metrics["answer_support_hit"] is True
    assert metrics["supporting_title_hit"] is True
    assert metrics["supporting_title_recall_at_k"] == 1.0
