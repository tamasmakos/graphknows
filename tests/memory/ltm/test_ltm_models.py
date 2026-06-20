"""Tests for LTM Pydantic models."""

from __future__ import annotations

import pytest

from graphknows.memory.ltm.models import CentralityResult, CommunityResult, GDSResult


def test_gds_result_defaults() -> None:
    r = GDSResult()
    assert r.entity_count == 0
    assert r.community_count == 0
    assert r.did_converge is False
    assert r.changed_community_ids == []
    assert r.modularities == []


def test_gds_result_populated() -> None:
    r = GDSResult(
        entity_count=10,
        edge_count=20,
        community_count=3,
        modularity=0.42,
        modularities=[0.40, 0.42],
        ran_iterations=2,
        did_converge=True,
        changed_community_ids=[1, 2],
        silhouette_cosine=0.75,
        global_separation=0.8,
        global_overlap=0.2,
        unweighted_modularity=0.38,
    )
    assert r.entity_count == 10
    assert r.community_count == 3
    assert r.did_converge is True
    assert r.silhouette_cosine == pytest.approx(0.75)


def test_community_result_defaults() -> None:
    r = CommunityResult()
    assert r.algorithm_used == ""
    assert r.error == ""


def test_community_result_populated() -> None:
    r = CommunityResult(
        community_count=5, modularity=0.5, algorithm_used="louvain", did_converge=True
    )
    assert r.community_count == 5
    assert r.algorithm_used == "louvain"


def test_centrality_result_defaults() -> None:
    r = CentralityResult()
    assert r.entity_count == 0
    assert r.error == ""


def test_centrality_result_error() -> None:
    r = CentralityResult(error="too_small:2")
    assert "too_small" in r.error


def test_gds_result_none_optional_fields() -> None:
    r = GDSResult()
    assert r.silhouette_cosine is None
    assert r.global_separation is None
    assert r.unweighted_modularity is None


def test_gds_result_model_dump() -> None:
    r = GDSResult(entity_count=5, community_count=2)
    d = r.model_dump()
    assert d["entity_count"] == 5
    assert "community_count" in d
