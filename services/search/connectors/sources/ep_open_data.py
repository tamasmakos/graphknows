"""EP Open Data Portal connector — European Parliament official documents API.

API docs: https://data.europarl.europa.eu/api/v2
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from ..base import ConnectorType, HttpMixin, SourceConnector
from ...models import Document, SearchQuery, SearchResult
from ...parsed_document import ParsedDocument

log = logging.getLogger(__name__)

_API_BASE = "https://data.europarl.europa.eu/api/v2"
_DOCS_ENDPOINT = f"{_API_BASE}/documents"
_VOTES_ENDPOINT = f"{_API_BASE}/votes"

# JSON-LD media type required by the EP Open Data API
_ACCEPT = "application/ld+json"


class EpOpenDataConnector(SourceConnector, HttpMixin):
    """Queries the EP Open Data Portal REST API (JSON-LD).

    Supports document search and vote records. Maps DCAT-AP/JSON-LD
    responses to the unified SearchResult / Document model.
    """

    source_id = "ep_open_data"
    connector_type = ConnectorType.REST_API
    rate_limit = 1.0
    institution = "European Parliament"

    def __init__(self) -> None:
        self._http_init(rate_limit_rps=1.0, timeout=20)
        self._doc_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # SourceConnector interface                                            #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "format": _ACCEPT,
            "limit": min(query.limit * 2, 50),
        }
        if query.date_from:
            params["dateFrom"] = query.date_from
        if query.date_to:
            params["dateTo"] = query.date_to

        client = self._ensure_client()
        results: list[SearchResult] = []
        q_lower = query.query.lower()

        for endpoint in (_DOCS_ENDPOINT, _VOTES_ENDPOINT):
            try:
                resp = await client.get(
                    endpoint,
                    params=params,
                    headers={"Accept": _ACCEPT},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.warning("[ep_open_data] API call failed %s: %s", endpoint, exc)
                continue

            items = self._extract_items(data)
            for item in items:
                title = self._item_title(item)
                if q_lower and q_lower not in title.lower():
                    continue
                doc_id = self._item_id(item)
                url = self._item_url(item)
                date = self._item_date(item)
                self._doc_cache[doc_id] = item
                results.append(
                    SearchResult(
                        id=doc_id,
                        source=self.source_id,
                        title=title,
                        date=date,
                        url=url,
                        excerpt=self._item_excerpt(item),
                    )
                )
                if len(results) >= query.limit:
                    return results

        return results

    async def fetch(self, document_id: str) -> Document:
        cached = self._doc_cache.get(document_id)
        if cached is not None:
            return self._item_to_document(cached)
        raise FileNotFoundError(f"[ep_open_data] not found: {document_id}")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_items(self, data: dict | list) -> list[dict]:
        if isinstance(data, list):
            return data
        for key in ("@graph", "results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return []

    def _item_title(self, item: dict) -> str:
        for key in ("dcterms:title", "schema:name", "title", "label"):
            v = item.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v:
                return str(v[0])
            if isinstance(v, dict):
                return str(next(iter(v.values()), ""))
        return "EP Document"

    def _item_id(self, item: dict) -> str:
        uri = item.get("@id") or item.get("id") or item.get("uri") or ""
        return self.canonical_id(str(uri))

    def _item_url(self, item: dict) -> str | None:
        for key in ("@id", "id", "uri", "url", "dcterms:identifier"):
            v = item.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
        return None

    def _item_date(self, item: dict) -> str | None:
        for key in ("dcterms:date", "schema:datePublished", "date", "timestamp"):
            v = item.get(key)
            if isinstance(v, str):
                return v[:10]
        return None

    def _item_excerpt(self, item: dict) -> str | None:
        for key in ("dcterms:description", "schema:description", "description", "abstract"):
            v = item.get(key)
            if isinstance(v, str):
                return v[:300]
            if isinstance(v, list) and v:
                return str(v[0])[:300]
        return None

    def _item_to_document(self, item: dict) -> Document:
        return Document(
            id=self._item_id(item),
            source=self.source_id,
            title=self._item_title(item),
            text=self._item_excerpt(item) or "",
            metadata={
                "url": self._item_url(item),
                "date": self._item_date(item),
                "institution": self.institution,
                "raw": item,
            },
            ingested_at=datetime.utcnow().isoformat(),
        )
