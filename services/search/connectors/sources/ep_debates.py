"""EP Debates connector — live queries to EP Open Data API /speeches endpoint."""
from __future__ import annotations

import json
import logging

from ..base import ConnectorType, HttpMixin, SourceConnector
from ...models import Document, SearchQuery, SearchResult

log = logging.getLogger(__name__)

_EP_API = "https://data.europarl.europa.eu/api/v2"



class EPDebatesConnector(SourceConnector, HttpMixin):
    """Live connector for EP plenary speech transcripts via the EP Open Data API."""

    source_id = "ep_debates"
    connector_type = ConnectorType.REST_API
    rate_limit = 1.0

    def __init__(self) -> None:
        self._http_init()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        try:
            raw = await self._get(
                f"{_EP_API}/speeches",
                params={"format": "application/ld+json", "limit": min(query.limit * 10, 200)},
            )
            data = json.loads(raw)
        except Exception as exc:
            log.warning("[ep_debates] API error: %s", exc)
            return []

        items = data.get("data") or data.get("@graph") or (data if isinstance(data, list) else [])
        q_lower = query.query.lower()
        results: list[SearchResult] = []
        for item in items:
            activity_id = item.get("activity_id", "")
            title = " ".join(activity_id.replace("-", " ").split()) if activity_id else "EP Activity"
            date = item.get("activity_date", "")[:10]
            item_uri = item.get("id", "")
            url = f"https://data.europarl.europa.eu/{item_uri}" if item_uri else None
            if q_lower and q_lower not in title.lower() and q_lower not in (item_uri or "").lower():
                continue
            if query.date_from and date and date < query.date_from:
                continue
            if query.date_to and date and date > query.date_to:
                continue
            results.append(
                SearchResult(
                    id=activity_id or self.canonical_id(item_uri or title),
                    source=self.source_id,
                    title=title,
                    date=date or None,
                    url=url,
                    excerpt=None,
                )
            )
            if len(results) >= query.limit:
                break
        return results

    async def fetch(self, document_id: str) -> Document:
        raise NotImplementedError(
            "[ep_debates] full document ingestion is handled by the scraper pipeline"
        )
