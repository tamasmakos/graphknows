"""The hand-labelled identity sample is well formed and scoreable (FR-040)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.identity import pairwise_scores

SAMPLE = Path(__file__).resolve().parents[2] / "evaluation/data/identity/labelled.jsonl"


@pytest.fixture(scope="module")
def clusters() -> list[dict[str, Any]]:
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class TestLabelledSample:
    def test_procedure_is_recorded_beside_the_file(self) -> None:
        assert (SAMPLE.parent / "README.md").exists()

    def test_sample_is_large_enough_to_score(self, clusters: list[dict[str, Any]]) -> None:
        assert len(clusters) >= 200

    def test_every_cluster_names_a_referent_and_merges_mentions(
        self, clusters: list[dict[str, Any]]
    ) -> None:
        assert all(
            cluster["referent"] and len(set(cluster["mentions"])) >= 2 for cluster in clusters
        )

    def test_no_mention_belongs_to_two_referents(self, clusters: list[dict[str, Any]]) -> None:
        mentions = [mention for cluster in clusters for mention in cluster["mentions"]]
        assert len(mentions) == len(set(mentions))

    def test_sample_scores_as_its_own_ground_truth(self, clusters: list[dict[str, Any]]) -> None:
        labelled = [cluster["mentions"] for cluster in clusters]
        score = pairwise_scores(labelled, labelled)
        assert (score.pairwise_precision, score.pairwise_recall) == (1.0, 1.0)
        assert score.n_clusters == len(clusters)
