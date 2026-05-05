"""
UK Legislation adapter — thin async wrapper around scrapers/uk_legislation.py.

The underlying scraper exposes module-level functions:
  - run_incremental(leg_types=None, days=1) -> list[dict]
  - fetch_document(leg_type, entry) -> dict | None

Each document dict has keys:
  doc_id, leg_type, title, text, url, updated, source, source_type, fetched_at, char_count
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import os

import scrapers.uk_legislation as _ukleg

from search.adapters.base import SourceAdapter
from search.models import Document, SearchQuery, SearchResult

_OUT_DIR = Path(os.environ.get("UK_LEGISLATION_OUT_DIR", "/data/raw/uk_legislation"))


class UKLegislationAdapter(SourceAdapter):
    """Wraps the UK Legislation scraper module."""

    source_id = "uk_legislation"
    rate_limit = 2.0

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        cached = self._load_cached()
        if not cached:
            cached = await asyncio.to_thread(_ukleg.run_incremental, days=2)

        q_lower = query.query.lower()
        results: list[SearchResult] = []
        for doc in cached:
            title = doc.get("title", "")
            text = doc.get("text", "")
            if q_lower not in title.lower() and q_lower not in text.lower():
                continue
            updated = doc.get("updated", "")
            if query.date_from and updated and updated < query.date_from:
                continue
            if query.date_to and updated and updated > query.date_to:
                continue

            results.append(SearchResult(
                id=self.canonical_id(doc),
                source=self.source_id,
                title=title,
                date=updated or None,
                url=doc.get("url"),
                excerpt=text[:300] if text else None,
            ))
            if len(results) >= query.limit:
                break

        return results

    async def fetch(self, document_id: str) -> Document:
        """
        document_id is a doc_id like 'uksi-2026-123'.
        Try cache first, then live fetch via the feed entry format.
        """
        raw = self._find_cached(document_id)
        if raw is None:
            raise FileNotFoundError(f"uk_legislation document not found: {document_id}")

        return Document(
            id=self.canonical_id(raw),
            source=self.source_id,
            title=raw.get("title", ""),
            text=raw.get("text", ""),
            metadata={
                "leg_type": raw.get("leg_type", ""),
                "source_type": raw.get("source_type", ""),
                "updated": raw.get("updated", ""),
            },
            ingested_at=raw.get("fetched_at", datetime.utcnow().isoformat()),
        )

    def canonical_id(self, raw: dict) -> str:
        return raw.get("doc_id", "") or raw.get("id", "")

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

    def _find_cached(self, doc_id: str) -> dict | None:
        p = _OUT_DIR / f"{doc_id}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        # Broader scan
        for p in _OUT_DIR.glob("*.json"):
            if p.name == "state.json":
                continue
            try:
                data = json.loads(p.read_text())
                if data.get("doc_id") == doc_id:
                    return data
            except Exception:
                pass
        return None
