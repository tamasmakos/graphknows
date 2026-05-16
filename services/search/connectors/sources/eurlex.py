"""EUR-Lex connector — live queries to EUR-Lex OJ L-series Atom feed."""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from xml.etree import ElementTree as ET

from ..base import ConnectorType, HttpMixin, SourceConnector
from ...models import Document, SearchQuery, SearchResult


log = logging.getLogger(__name__)

_OJ_HTML = (
    "https://eur-lex.europa.eu/oj/direct-access.html"
    "?ojType=L&ojDate={date}"
)
_EURLEX_HTML = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:{celex}&from=EN"
)
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_TARGET_TYPES = {"R", "L", "D"}  # Regulation, Directive, Decision
_CELEX_RE = re.compile(r"uri=(?:CELEX|celex)(?:%3A|:)([0-9A-Za-z]+)", re.IGNORECASE)


class EurLexConnector(SourceConnector, HttpMixin):
    """Live connector for EU legislation via the EUR-Lex OJ L-series Atom feed."""

    source_id = "eurlex"
    connector_type = ConnectorType.RSS_FEED
    rate_limit = 1.0

    def __init__(self) -> None:
        self._http_init()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        q_lower = query.query.lower()
        results: list[SearchResult] = []
        seen: set[str] = set()
        for days_back in range(30):
            if len(results) >= query.limit:
                break
            day = (date.today() - timedelta(days=days_back)).isoformat()
            try:
                html = await self._get(_OJ_HTML.format(date=day))
                entries = _parse_oj_html(html, day)
            except Exception:
                continue
            for entry in entries:
                if len(results) >= query.limit:
                    break
                celex = entry["celex"]
                if celex in seen:
                    continue
                seen.add(celex)
                if _celex_type(celex) not in _TARGET_TYPES:
                    continue
                pub_date = entry["published"]
                title = celex  # CELEX is the primary identifier; title not available from listing
                if q_lower and q_lower not in title.lower():
                    continue
                if query.date_from and pub_date and pub_date < query.date_from:
                    continue
                if query.date_to and pub_date and pub_date > query.date_to:
                    continue
                results.append(
                    SearchResult(
                        id=celex,
                        source=self.source_id,
                        title=title,
                        date=pub_date or None,
                        url=f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}",
                        excerpt=None,
                    )
                )
        return results

    async def fetch(self, document_id: str) -> Document:
        url = _EURLEX_HTML.format(celex=document_id)
        try:
            html = await self._get(url)
        except Exception as exc:
            raise FileNotFoundError(f"[eurlex] CELEX not found: {document_id}") from exc
        text = _strip_html(html)
        import datetime as _dt
        return Document(
            id=document_id,
            source=self.source_id,
            title=document_id,
            text=text,
            metadata={"url": url},
            ingested_at=_dt.datetime.utcnow().isoformat(),
        )


def _parse_oj_html(html: str, day: str) -> list[dict]:
    """Extract CELEX identifiers from EUR-Lex OJ daily-view HTML page."""
    return [
        {"celex": m, "published": day}
        for m in dict.fromkeys(_CELEX_RE.findall(html))  # preserve order, deduplicate
    ]


def _celex_type(celex: str) -> str | None:
    m = re.match(r"[0-9]\d{4}([A-Z])\d+", celex)
    return m.group(1) if m else None


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:50_000]
