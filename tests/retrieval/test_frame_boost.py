"""Frame re-ranking: the scoring policy, in isolation from any store.

Exists because the measured failure on conv-30 is order, not recall — gold sits
in the pool (probe 0.948) and does not survive the cut (context 0.765).
"""

from __future__ import annotations

import pytest

from graphknows.ranking import frame_boost


def test_a_frame_in_every_chunk_contributes_nothing() -> None:
    # Self_motion covered 54 of conv-30's 110 chunks; a frame in ALL of them is
    # pure noise and must not tip anything.
    assert frame_boost.idf(df=110, total=110) == 0.0
    assert frame_boost.idf(df=2, total=110) > frame_boost.idf(df=50, total=110)


def test_saturated_frames_never_demote() -> None:
    # log of a ratio below 1 is negative; without the floor a common frame would
    # actively push its chunk down.
    assert frame_boost.idf(df=200, total=110) == 0.0


def test_role_binding_outweighs_a_bare_frame_match() -> None:
    df, total = {"Giving": 4}, 100
    bare = frame_boost.score_chunks(
        [{"chunk_id": "c1", "frame": "Giving", "anchor_hits": 0}], df, total
    )
    bound = frame_boost.score_chunks(
        [{"chunk_id": "c1", "frame": "Giving", "anchor_hits": 1}], df, total
    )
    assert bound["c1"] == pytest.approx(bare["c1"] * frame_boost.ROLE_BINDING_WEIGHT)


def test_scores_accumulate_over_matching_instances() -> None:
    rows = [
        {"chunk_id": "c1", "frame": "Giving", "anchor_hits": 0},
        {"chunk_id": "c1", "frame": "Death", "anchor_hits": 0},
        {"chunk_id": "c2", "frame": "Giving", "anchor_hits": 0},
    ]
    scores = frame_boost.score_chunks(rows, {"Giving": 4, "Death": 4}, 100)
    assert scores["c1"] > scores["c2"]


def test_a_frame_match_can_lift_a_chunk_over_one_ranked_above_it() -> None:
    # RRF at the depths that matter: rank ~25 scores 0.0118, rank ~40 0.0100.
    combined = {"gold": (0.0100, {"sources": {"bm25"}}), "other": (0.0118, {"sources": {"vector"}})}
    out = frame_boost.apply(combined, {"gold": 2.0}, alpha=0.5)
    assert out["gold"][0] > out["other"][0]


def test_chunks_without_a_frame_signal_keep_their_score_exactly() -> None:
    combined = {"a": (0.02, {"sources": {"bm25"}}), "b": (0.01, {"sources": {"bm25"}})}
    out = frame_boost.apply(combined, {"a": 1.0}, alpha=0.5)
    assert out["b"][0] == 0.01  # only ever promotes
    assert out["a"][0] > 0.02


def test_the_boost_is_marked_as_a_source() -> None:
    out = frame_boost.apply({"a": (0.01, {"sources": {"bm25"}})}, {"a": 1.0})
    assert "frame_rank" in out["a"][1]["sources"]


def test_no_signal_leaves_the_pool_untouched() -> None:
    combined = {"a": (0.02, {"sources": {"bm25"}})}
    assert frame_boost.apply(combined, {}) is combined
    assert frame_boost.apply(combined, {"a": 0.0}) is combined


def test_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_FRAME_RERANK", "false")
    assert frame_boost.enabled() is False
    monkeypatch.setenv("GRAPHKNOWS_FRAME_RERANK", "true")
    assert frame_boost.enabled() is True
