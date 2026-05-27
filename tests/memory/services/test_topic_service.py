"""Tests for TopicService — uses AsyncMock for IGraphDb and ILlm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.services.topic_service import TopicService


@pytest.fixture()
def mock_graph_db() -> MagicMock:
    db = MagicMock()
    db.summarize_topics = AsyncMock(return_value=3)
    return db


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = MagicMock(return_value='{"title": "AI Research", "summary": "..."}')
    return llm


@pytest.fixture()
def service(mock_graph_db: MagicMock, mock_llm: MagicMock) -> TopicService:
    return TopicService(graph_db=mock_graph_db, llm=mock_llm)


@pytest.mark.asyncio
async def test_summarize_topics_delegates(
    service: TopicService, mock_graph_db: MagicMock
) -> None:
    result = await service.summarize_topics("sess-1", "key-123")
    mock_graph_db.summarize_topics.assert_awaited_once_with(
        "sess-1", "key-123", "meta-llama/llama-3.1-8b-instruct", None
    )
    assert result == 3


@pytest.mark.asyncio
async def test_summarize_topics_with_pg_store(
    service: TopicService, mock_graph_db: MagicMock
) -> None:
    fake_pg = MagicMock()
    result = await service.summarize_topics(
        "sess-2", "key-456", "custom/model", pg_store=fake_pg
    )
    mock_graph_db.summarize_topics.assert_awaited_once_with(
        "sess-2", "key-456", "custom/model", fake_pg
    )
    assert result == 3


@pytest.mark.asyncio
async def test_summarize_topics_zero_topics(
    service: TopicService, mock_graph_db: MagicMock
) -> None:
    mock_graph_db.summarize_topics = AsyncMock(return_value=0)
    result = await service.summarize_topics("sess-empty", "key")
    assert result == 0
