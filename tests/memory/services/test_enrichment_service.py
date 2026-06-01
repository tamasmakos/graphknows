"""Tests for EnrichmentService — uses AsyncMock for IGraphDb and ILlm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.services.enrichment_service import EnrichmentService


@pytest.fixture()
def mock_graph_db() -> MagicMock:
    db = MagicMock()
    db.run_gds_enrichment = AsyncMock(
        return_value={
            "entity_count": 42,
            "edge_count": 100,
            "community_count": 5,
            "modularity": 0.72,
        }
    )
    return db


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = MagicMock(return_value="summary text")
    return llm


@pytest.fixture()
def service(mock_graph_db: MagicMock, mock_llm: MagicMock) -> EnrichmentService:
    return EnrichmentService(graph_db=mock_graph_db, llm=mock_llm)


@pytest.mark.asyncio
async def test_run_gds_enrichment_delegates(
    service: EnrichmentService, mock_graph_db: MagicMock
) -> None:
    result = await service.run_gds_enrichment("sess-abc")
    mock_graph_db.run_gds_enrichment.assert_awaited_once_with("sess-abc")
    assert result["entity_count"] == 42
    assert result["community_count"] == 5


@pytest.mark.asyncio
async def test_run_gds_enrichment_propagates_error(
    service: EnrichmentService, mock_graph_db: MagicMock
) -> None:
    mock_graph_db.run_gds_enrichment = AsyncMock(side_effect=RuntimeError("GDS failure"))
    with pytest.raises(RuntimeError, match="GDS failure"):
        await service.run_gds_enrichment("sess-err")
