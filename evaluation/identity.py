"""Pairwise identity precision and recall over a hand-labelled sample (FR-040).

Entity resolution is scored the way a clustering is scored: not on cluster names,
which are arbitrary, but on the pairs of mentions a clustering puts together.
Precision is the share of merged pairs the labels agree with; recall the share of
labelled pairs the resolver actually found. ``dataclasses.asdict`` on the result
is the contract's ``identity`` object (contracts/panel-report.md).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class IdentityScore:
    """One scoring of a predicted clustering against the labelled one.

    Both scores are 0.0 when their denominator is empty — a clustering with no
    merges at all has no precision to speak of, and is reported as such rather
    than credited with a perfect one.
    """

    pairwise_precision: float
    pairwise_recall: float
    n_clusters: int


def _co_membership_pairs(clusters: Sequence[Collection[str]]) -> set[frozenset[str]]:
    """Every unordered pair of mentions that *clusters* places in one cluster."""
    return {
        frozenset(pair) for cluster in clusters for pair in combinations(sorted(set(cluster)), 2)
    }


def pairwise_scores(
    labelled: Sequence[Collection[str]],
    predicted: Sequence[Collection[str]],
) -> IdentityScore:
    """Score *predicted* clusters of mentions against the hand-*labelled* ones."""
    labelled_pairs = _co_membership_pairs(labelled)
    predicted_pairs = _co_membership_pairs(predicted)
    agreed = len(labelled_pairs & predicted_pairs)
    return IdentityScore(
        pairwise_precision=agreed / len(predicted_pairs) if predicted_pairs else 0.0,
        pairwise_recall=agreed / len(labelled_pairs) if labelled_pairs else 0.0,
        n_clusters=len(labelled),
    )
