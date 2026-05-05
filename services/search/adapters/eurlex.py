"""
EUR-Lex adapter — thin async wrapper around scrapers/eurlex.py.

The underlying scraper exposes module-level functions:
  - run_incremental(days=1) -> list[dict]
  - fetch_celex(celex, title, published, source_url) -> dict | None

Each document dict has keys:
  celex, title, text, url, published, source, source_type, fetched_at, char_count
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import os

import scrapers.eurlex as _eurlex

from search.adapters.base import SourceAdapter
from search.models import Document, SearchQuery, SearchResult

_OUT_DIR = Path(os.environ.get("EURLEX_OUT_DIR", "/data/raw/eurlex"))


class EurLexAdapter(SourceAdapter):
    """Wraps the EUR-Lex legislation scraper module."""

    source_id = "eurlex"
    rate_limit = 1.0

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        cached = self._load_cached()
        if not cached:
            cached = await asyncio.to_thread(_eurlex.run_incremental, days=2)

        q_lower = query.query.lower()
        results: list[SearchResult] = []
        for doc in cached:
            title = doc.get("title", "")
            text = doc.get("text", "")
            if q_lower not in title.lower() and q_lower not in text.lower():
                continue
            published = doc.get("published", "")
            if query.date_from and published and published < query.date_from:
                continue
            if query.date_to and published and published > query.date_to:
                continue

            results.append(SearchResult(
                id=self.canonical_id(doc),
                source=self.source_id,
                title=title,
                date=published or None,
                url=doc.get("url"),
                excerpt=text[:300] if text else None,
            ))
            if len(results) >= query.limit:
                break

        return results

    async def fetch(self, document_id: str) -> Document:
        """
        document_id is a CELEX number, e.g. '32023R0001'.
        Try cache first, then live fetch.
        """
        raw = self._find_cached(document_id)
        if raw is None:
            # Live fetch from EUR-Lex
            raw = await asyncio.to_thread(_eurlex.fetch_celex, document_id, "", "", "")
        if raw is None:
            raise FileNotFoundError(f"eurlex document not found: {document_id}")

        return Document(
            id=self.canonical_id(raw),
            source=self.source_id,
            title=raw.get("title", ""),
            text=raw.get("text", ""),
            metadata={
                "celex": raw.get("celex", document_id),
                "doc_type_code": raw.get("doc_type_code", ""),
                "source_type": raw.get("source_type", ""),
                "published": raw.get("published", ""),
            },
            ingested_at=raw.get("fetched_at", datetime.utcnow().isoformat()),
        )

    def canonical_id(self, raw: dict) -> str:
        return raw.get("celex", "") or raw.get("id", "")

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _load_cached(self) -> list[dict]:
        if not _OUT_DIR.exists():
            return []
        docs = []
        for p in _OUT_DIR.glob("*.json"):
            if p.name == "state.json":
                continue
            try:
                docs.append(json.loads(p.read_text()))
            except Exception:
                pass
        return docs

    def _find_cached(self, celex: str) -> dict | None:
        p = _OUT_DIR / f"{celex}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return None
