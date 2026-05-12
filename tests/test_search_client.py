"""Tests for SearchServiceClient — all calls mocked at the httpx level."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from services.agent.tools.search_client import SearchServiceClient
from services.search.models import ScrapeJob, SearchResult


@pytest.fixture
def mock_http_client() -> MagicMock:
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def client(mock_http_client: MagicMock) -> SearchServiceClient:
    return SearchServiceClient(
        base_url="http://test-search:8002",
        http_client=mock_http_client,  # type: ignore[arg-type]
    )


class TestSearch:
    async def test_search_returns_parsed_results(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"id": "doc1", "source": "eurlex", "title": "Doc 1", "date": "2024-01-01", "url": "http://example.com/1", "excerpt": "Excerpt 1"},
            {"id": "doc2", "source": "eurlex", "title": "Doc 2", "date": "2024-01-02", "url": "http://example.com/2", "excerpt": "Excerpt 2"},
        ]
        mock_http_client.post = AsyncMock(return_value=mock_response)

        results = await client.search(query="test query", sources=["eurlex"], limit=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].id == "doc1"
        assert results[1].source == "eurlex"
        mock_http_client.post.assert_awaited_once()

    async def test_search_empty_results(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = []
        mock_http_client.post = AsyncMock(return_value=mock_response)

        results = await client.search(query="nonexistent", limit=10)
        assert results == []

    async def test_search_raises_on_http_error(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_http_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock(status_code=500),
        ))
        with pytest.raises(httpx.HTTPStatusError):
            await client.search(query="fail")


class TestScrape:
    async def test_scrape_returns_job(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "job_id": "job-123", "status": "queued", "created_at": "2024-01-01T00:00:00Z",
            "progress": 0.0, "result": None, "error": None,
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)

        job = await client.scrape(source="eurlex", document_ids=["doc1"])
        assert isinstance(job, ScrapeJob)
        assert job.job_id == "job-123"
        assert job.status == "queued"


class TestPollJob:
    async def test_poll_until_complete(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        responses = [
            MagicMock(spec=httpx.Response, json=lambda: {"job_id": "j1", "status": "running", "created_at": "", "progress": 0.5, "result": None, "error": None}),
            MagicMock(spec=httpx.Response, json=lambda: {"job_id": "j1", "status": "complete", "created_at": "", "progress": 1.0, "result": [{"id": "d1"}], "error": None}),
        ]
        mock_http_client.get = AsyncMock(side_effect=responses)

        job = await client.poll_job("j1", timeout=5.0, interval=0.05)
        assert job.status == "complete"
        assert job.result == [{"id": "d1"}]
        assert mock_http_client.get.await_count == 2

    async def test_poll_raises_timeout(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"job_id": "j1", "status": "running", "created_at": "", "progress": 0.0, "result": None, "error": None}
        mock_http_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(TimeoutError, match="did not complete"):
            await client.poll_job("j1", timeout=0.1, interval=0.05)

    async def test_poll_returns_failed_job(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"job_id": "j1", "status": "failed", "created_at": "", "progress": 0.5, "result": None, "error": "something went wrong"}
        mock_http_client.get = AsyncMock(return_value=mock_response)

        job = await client.poll_job("j1")
        assert job.status == "failed"
        assert job.error == "something went wrong"


class TestFetchDocuments:
    async def test_fetch_documents_end_to_end(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        search_resp = MagicMock(spec=httpx.Response)
        search_resp.json.return_value = [
            {"id": "d1", "source": "eurlex", "title": "Doc 1", "date": "2024-01-01", "url": "http://ex.com/1", "excerpt": "Excerpt 1"},
        ]
        scrape_resp = MagicMock(spec=httpx.Response)
        scrape_resp.json.return_value = {"job_id": "j1", "status": "queued", "created_at": "", "progress": 0.0, "result": None, "error": None}
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"job_id": "j1", "status": "complete", "created_at": "", "progress": 1.0, "result": [{"id": "d1", "source": "eurlex", "title": "Doc 1", "text": "Full text here", "metadata": {"url": "http://ex.com/1"}, "ingested_at": "2024-01-01T00:00:00Z"}], "error": None}

        mock_http_client.post = AsyncMock(side_effect=[search_resp, scrape_resp])
        mock_http_client.get = AsyncMock(return_value=poll_resp)

        raw_docs = await client.fetch_documents(query="test", sources=["eurlex"], limit=5)
        assert len(raw_docs) == 1
        assert raw_docs[0].id == "d1"
        assert raw_docs[0].text == "Full text here"
        assert raw_docs[0].url == "http://ex.com/1"

    async def test_fetch_documents_no_results(self, client: SearchServiceClient, mock_http_client: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = []
        mock_http_client.post = AsyncMock(return_value=mock_response)

        raw_docs = await client.fetch_documents(query="nothing", limit=10)
        assert raw_docs == []
