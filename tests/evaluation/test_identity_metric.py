"""Identity is scored on pairs of mentions, not on cluster names (FR-040)."""

from __future__ import annotations

from dataclasses import asdict

from evaluation.identity import pairwise_scores

LABELLED = [{"a", "b", "c"}, {"d", "e"}]  # 3 + 1 = 4 labelled pairs


class TestPairwiseScores:
    def test_cluster_names_do_not_matter(self) -> None:
        score = pairwise_scores(LABELLED, [{"d", "e"}, {"c", "b", "a"}])
        assert (score.pairwise_precision, score.pairwise_recall) == (1.0, 1.0)

    def test_over_merging_costs_precision_only(self) -> None:
        score = pairwise_scores(LABELLED, [{"a", "b", "c", "d", "e"}])
        assert score.pairwise_precision == 4 / 10
        assert score.pairwise_recall == 1.0

    def test_splitting_a_cluster_costs_recall_only(self) -> None:
        score = pairwise_scores(LABELLED, [{"a", "b"}, {"c"}, {"d", "e"}])
        assert score.pairwise_precision == 1.0
        assert score.pairwise_recall == 2 / 4

    def test_no_merges_scores_zero_rather_than_a_free_perfect(self) -> None:
        score = pairwise_scores(LABELLED, [{"a"}, {"b"}, {"c"}, {"d"}, {"e"}])
        assert (score.pairwise_precision, score.pairwise_recall) == (0.0, 0.0)

    def test_record_matches_the_report_contract(self) -> None:
        assert asdict(pairwise_scores(LABELLED, LABELLED)) == {
            "pairwise_precision": 1.0,
            "pairwise_recall": 1.0,
            "n_clusters": 2,
        }
