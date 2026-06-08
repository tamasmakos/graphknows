"""Tests for LTMEnrichmentService with mocked Neo4j driver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.ltm.enrichment import LTMEnrichmentService
from graphknows.memory.ltm.models import CentralityResult, CommunityResult, GDSResult


def _make_driver(entity_count: int = 5, edge_count: int = 3) -> MagicMock:
    """Return a mock Neo4j driver with minimal session/run support."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    ec_row = MagicMock()
    ec_row.__getitem__ = MagicMock(side_effect=lambda k: entity_count if k == "c" else 0)
    edge_row = MagicMock()
    edge_row.__getitem__ = MagicMock(side_effect=lambda k: edge_count if k == "c" else 0)

    def run_side_effect(query: str, **_kwargs: object) -> MagicMock:
        result = MagicMock()
        if "count(e)" in query:
            result.single.return_value = ec_row
        elif "count(r)" in query:
            result.single.return_value = edge_row
        elif "louvain.write" in query:
            comm = MagicMock()
            comm.__getitem__ = MagicMock(side_effect=lambda k: {
                "communityCount": 2, "modularity": 0.5,
                "modularities": [0.5], "ranLevels": 1,
            }.get(k, 0))
            result.single.return_value = comm
        else:
            result.single.return_value = None
            result.data.return_value = []
        return result

    session.run.side_effect = run_side_effect
    return driver


def test_run_gds_too_small() -> None:
    driver = _make_driver(entity_count=2)
    svc = LTMEnrichmentService()
    result = svc.run_gds(driver)
    assert isinstance(result, GDSResult)
    assert "too_small" in result.error


def test_run_gds_returns_gds_result() -> None:
    driver = _make_driver(entity_count=5)
    svc = LTMEnrichmentService()
    with patch("graphknows.memory.ltm.enrichment.run_feature_engineering", return_value={
        "entity_count": 5, "edge_count": 3, "community_count": 2,
        "modularity": 0.5, "modularities": [0.5], "ran_iterations": 1,
        "did_converge": True, "changed_community_ids": [], "error": "",
    }) as mock_fe:
        result = svc.run_gds(driver)
        mock_fe.assert_called_once_with(driver)
    assert isinstance(result, GDSResult)
    assert result.entity_count == 5
    assert result.community_count == 2
    assert result.did_converge is True


def test_run_gds_skips_when_gds_plugin_is_unavailable() -> None:
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def run_side_effect(query: str, **_kwargs: object) -> MagicMock:
        result = MagicMock()
        row = MagicMock()
        if "count(e)" in query:
            row.__getitem__ = MagicMock(return_value=5)
        elif "count(r)" in query:
            row.__getitem__ = MagicMock(return_value=3)
        elif "SHOW PROCEDURES" in query:
            row.__getitem__ = MagicMock(return_value=0)
        result.single.return_value = row
        return result

    session.run.side_effect = run_side_effect

    result = LTMEnrichmentService().run_gds(driver)

    assert result.error == "gds_unavailable"
    assert result.entity_count == 5
    assert result.edge_count == 3


def test_run_community_detection_too_small() -> None:
    driver = _make_driver(entity_count=1)
    svc = LTMEnrichmentService()
    result = svc.run_community_detection(driver)
    assert isinstance(result, CommunityResult)
    assert "too_small" in result.error


def test_run_centrality_too_small() -> None:
    driver = _make_driver(entity_count=2)
    svc = LTMEnrichmentService()
    result = svc.run_centrality(driver)
    assert isinstance(result, CentralityResult)
    assert "too_small" in result.error


@pytest.mark.asyncio
async def test_run_topic_summarization_with_store() -> None:
    neo_store = MagicMock()
    neo_store.summarize_topics = AsyncMock(return_value=3)
    fake_pg = MagicMock()
    svc = LTMEnrichmentService()
    count = await svc.run_topic_summarization(
        session_id="sess-1",
        driver=neo_store,
        pg_store=fake_pg,
        llm_key="key",
        llm_model="model",
        gds_result=GDSResult(modularity=0.42, algorithm_used="leiden"),
    )
    assert count == 3
    neo_store.summarize_topics.assert_called_once_with(
        session_id="sess-1",
        openrouter_key="key",
        llm_model="model",
        pg_store=fake_pg,
        gds_modularity=0.42,
        community_algorithm="leiden",
    )


@pytest.mark.asyncio
async def test_run_topic_summarization_bare_driver_returns_zero() -> None:
    driver = _make_driver()
    driver.summarize_topics = AsyncMock(return_value=0)
    svc = LTMEnrichmentService()
    count = await svc.run_topic_summarization(
        session_id="sess-2",
        driver=driver,
        pg_store=MagicMock(),
        llm_key="key",
        llm_model="model",
    )
    assert count == 0


@pytest.mark.asyncio
async def test_run_gds_async() -> None:
    svc = LTMEnrichmentService()
    with patch.object(svc, "run_gds", return_value=GDSResult(entity_count=7)) as mock_run:
        result = await svc.run_gds_async(MagicMock())
        mock_run.assert_called_once()
    assert result.entity_count == 7
