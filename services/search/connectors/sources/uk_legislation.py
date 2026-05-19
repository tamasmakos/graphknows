"""UK Legislation connector — live queries to legislation.gov.uk Atom feeds."""

from __future__ import annotations

import logging
from xml.etree import ElementTree as ET

from ..base import ConnectorType, HttpMixin, SourceConnector
from ...models import Document, SearchQuery, SearchResult

log = logging.getLogger(__name__)

_FEEDS = [
    "https://www.legislation.gov.uk/uksi/data.feed?results-count=25",
    "https://www.legislation.gov.uk/ukpga/data.feed?results-count=25",
]
_NS = {"atom": "http://www.w3.org/2005/Atom"}


class UKLegislationConnector(SourceConnector, HttpMixin):
    """Live connector for UK legislation via the legislation.gov.uk Atom feeds."""

    source_id = "uk_legislation"
    connector_type = ConnectorType.RSS_FEED
    rate_limit = 1.0

    def __init__(self) -> None:
        self._http_init()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        q_lower = query.query.lower()
        results: list[SearchResult] = []
        for feed_url in _FEEDS:
            if len(results) >= query.limit:
                break
            try:
                xml_text = await self._get(feed_url)
                entries = _parse_atom(xml_text.encode())
            except Exception as exc:
                log.warning("[uk_legislation] feed error %s: %s", feed_url, exc)
                continue
            for entry in entries:
                if len(results) >= query.limit:
                    break
                title = entry.get("title", "")
                updated = entry.get("updated", "")[:10]
                url = entry.get("url", "")
                if q_lower and q_lower not in title.lower():
                    continue
                if query.date_from and updated and updated < query.date_from:
                    continue
                if query.date_to and updated and updated > query.date_to:
                    continue
                results.append(
                    SearchResult(
                        id=self.canonical_id(url or title),
                        source=self.source_id,
                        title=title,
                        date=updated or None,
                        url=url or None,
                        excerpt=None,
                    )
                )
        return results

    async def fetch(self, document_id: str) -> Document:
        raise NotImplementedError(
            "[uk_legislation] full document ingestion is handled by the scraper pipeline"
        )


def _parse_atom(xml_bytes: bytes) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    entries = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        link_el = entry.find("atom:link[@rel='alternate']", _NS)
        updated_el = entry.find("atom:updated", _NS)
        title = (title_el.text or "").strip() if title_el is not None else ""
        url = link_el.get("href", "") if link_el is not None else ""
        updated = (updated_el.text or "")[:10] if updated_el is not None else ""
        entries.append({"title": title, "url": url, "updated": updated})
    return entries
