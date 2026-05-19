"""data.europa.eu hub-search connector — EU open data portal aggregator.

API: https://data.europa.eu/api/hub/search/search
Covers datasets, distributions, and catalogues from EU open data hub.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..base import ConnectorType, HttpMixin, SourceConnector
from ...models import Document, SearchQuery, SearchResult

log = logging.getLogger(__name__)

_SEARCH_API = "https://data.europa.eu/api/hub/search/search"
_PORTAL_BASE = "https://data.europa.eu/data/datasets"


class DataEuropaConnector(SourceConnector, HttpMixin):
    """Queries the data.europa.eu hub-search API.

    Returns metadata-level results (dataset cards) using the DCAT-AP
    JSON format served by the portal. Best for locating open datasets
    from EU institutions, agencies, and member state portals.
    """

    source_id = "data_europa"
    connector_type = ConnectorType.REST_API
    rate_limit = 2.0
    institution = "data.europa.eu"

    def __init__(self) -> None:
        self._http_init(rate_limit_rps=2.0, timeout=15)
        self._doc_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # SourceConnector interface                                            #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "q": query.query,
            "filter": "dataset",
            "limit": min(query.limit, 50),
            "page": 1,
        }
        if query.date_from:
            params["issued_from"] = query.date_from
        if query.date_to:
            params["issued_to"] = query.date_to

        client = self._ensure_client()
        try:
            resp = await client.get(
                _SEARCH_API,
                params=params,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("[data_europa] search API failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for item in self._extract_results(data):
            doc_id = self._item_id(item)
            self._doc_cache[doc_id] = item
            results.append(
                SearchResult(
                    id=doc_id,
                    source=self.source_id,
                    title=self._item_title(item),
                    date=self._item_date(item),
                    url=self._item_url(item),
                    excerpt=self._item_excerpt(item),
                )
            )
            if len(results) >= query.limit:
                break
        return results

    async def fetch(self, document_id: str) -> Document:
        cached = self._doc_cache.get(document_id)
        if cached is not None:
            return self._item_to_document(cached)
        raise FileNotFoundError(f"[data_europa] not found: {document_id}")

    # ------------------------------------------------------------------ #
    # Private helpers — DCAT-AP field mapping                             #
    # ------------------------------------------------------------------ #

    def _extract_results(self, data: dict) -> list[dict]:
        # API returns {"result": {"results": [...]}}
        try:
            return data["result"]["results"]
        except (KeyError, TypeError):
            pass
        for key in ("results", "items", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []

    def _item_title(self, item: dict) -> str:
        title = item.get("title") or item.get("label") or {}
        if isinstance(title, dict):
            return title.get("en") or next(iter(title.values()), "EU Dataset")
        return str(title) or "EU Dataset"

    def _item_id(self, item: dict) -> str:
        raw_id = item.get("id") or item.get("identifier") or item.get("@id") or ""
        return self.canonical_id(str(raw_id))

    def _item_url(self, item: dict) -> str | None:
        # Try landing page, then construct from id
        lp = item.get("landingPage") or item.get("landing_page")
        if isinstance(lp, list) and lp:
            lp = lp[0]
        if isinstance(lp, str) and lp.startswith("http"):
            return lp
        raw_id = item.get("id") or item.get("identifier") or ""
        if raw_id:
            return f"{_PORTAL_BASE}/{raw_id}"
        return None

    def _item_date(self, item: dict) -> str | None:
        for key in ("issued", "modified", "temporal", "releaseDate"):
            v = item.get(key)
            if isinstance(v, str):
                return v[:10]
        return None

    def _item_excerpt(self, item: dict) -> str | None:
        desc = item.get("description") or item.get("notes") or {}
        if isinstance(desc, dict):
            text = desc.get("en") or next(iter(desc.values()), "")
        else:
            text = str(desc)
        return text[:300] if text else None

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
            },
            ingested_at=datetime.utcnow().isoformat(),
        )
