"""Tests for RelationService — uses AsyncMock for IGraphDb and ILlm."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.services.relation_service import RelationService


@pytest.fixture()
def mock_graph_db() -> MagicMock:
    db = MagicMock()
    db.extract_and_write_relations = AsyncMock(return_value=7)
    return db


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = MagicMock(return_value="REGULATES")
    return llm


@pytest.fixture()
def service(mock_graph_db: MagicMock, mock_llm: MagicMock) -> RelationService:
    return RelationService(graph_db=mock_graph_db, llm=mock_llm)


@pytest.fixture()
def stm_data() -> dict[str, Any]:
    return {
        "chunks": [{"id": "c1", "text": "Alice regulates Bob."}],
        "entities": [{"name": "Alice"}, {"name": "Bob"}],
        "chunk_entity_edges": [
            {"chunk_id": "c1", "entity_name": "Alice"},
            {"chunk_id": "c1", "entity_name": "Bob"},
        ],
    }


@pytest.mark.asyncio
async def test_extract_and_write_relations_delegates(
    service: RelationService,
    mock_graph_db: MagicMock,
    stm_data: dict[str, Any],
) -> None:
    result = await service.extract_and_write_relations(stm_data, "openrouter-key")
    mock_graph_db.extract_and_write_relations.assert_awaited_once_with(
        stm_data,
        "openrouter-key",
        "openrouter/meta-llama/llama-3.1-8b-instruct",
    )
    assert result == 7


@pytest.mark.asyncio
async def test_extract_and_write_relations_custom_model(
    service: RelationService,
    mock_graph_db: MagicMock,
    stm_data: dict[str, Any],
) -> None:
    result = await service.extract_and_write_relations(
        stm_data, "key", "openrouter/custom/model"
    )
    mock_graph_db.extract_and_write_relations.assert_awaited_once_with(
        stm_data, "key", "openrouter/custom/model"
    )
    assert result == 7


@pytest.mark.asyncio
async def test_extract_and_write_relations_propagates_error(
    service: RelationService,
    mock_graph_db: MagicMock,
    stm_data: dict[str, Any],
) -> None:
    mock_graph_db.extract_and_write_relations = AsyncMock(
        side_effect=ValueError("extraction failed")
    )
    with pytest.raises(ValueError, match="extraction failed"):
        await service.extract_and_write_relations(stm_data, "key")
