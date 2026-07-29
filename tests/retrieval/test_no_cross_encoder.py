"""The fused RRF score IS the ranking. There is no cross-encoder.

`cross-encoder/ms-marco-MiniLM-L6-v2` was the pipeline's sole ranker, introduced
by structural deduction and never A/B'd. Ranking is the ENTIRE retrieval deficit
here -- the candidate pool's ceiling is 1.000 evidence_recall at full depth -- so
the ranker was measured directly, same namespace, same channels, same pool, with
the cross-encoder replaced by the identity function:

    arm                        @5      @10      @25      @50     @100    secs
    A cross-encoder         0.368    0.433    0.567    0.687    0.827     178
    B fused order (no CE)   0.560    0.622    0.692    0.773    0.819      51
    delta (A - B)          -0.192   -0.190   -0.126   -0.086   +0.008    +127

It was WORSE at every depth a caller actually uses, reaching parity only at
k=100 where the ordering barely matters, and it cost 3.5x the wall clock to do
it. Deleting it is worth +0.125 evidence_recall@25.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_rerank_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graphknows.ranking.rerank")


def test_retriever_does_not_reference_a_cross_encoder() -> None:
    from graphknows.retrieval import retriever

    assert not hasattr(retriever, "_ce_rerank"), (
        "a re-introduced reranker must come with an A/B, not a deduction"
    )
