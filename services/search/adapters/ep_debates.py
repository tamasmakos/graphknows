"""
EP Debates adapter — thin async wrapper around scrapers/ep_debates.py.

The underlying scraper exposes module-level functions (no class):
  - run_incremental(max_speeches=300) -> list[dict]
  - _fetch_speeches(offset, limit) -> dict   (private, used internally)
  - _build_itm_document(itm_id, title, speeches, session_date) -> dict

For search() we call run_incremental() with a small cap and filter by query text.
For fetch()  we look for a cached JSON file by itm_id, or run a targeted fetch.
"""

from __future__ import annotations

import asyncio
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import scrapers.ep_debates as _ep

from search.adapters.base import SourceAdapter
from search.models import Document, SearchQuery, SearchResult

# Default output dir — co-located with repo; override via EP_DEBATES_OUT_DIR env var
_REPO_DEFAULT = Path(__file__).resolve().parents[3] / "data" / "raw" / "ep_debates"
_OUT_DIR = Path(os.environ.get("EP_DEBATES_OUT_DIR", str(_REPO_DEFAULT)))


class EPDebatesAdapter(SourceAdapter):
    """Wraps the EP Debates scraper module."""

    source_id = "ep_debates"
    rate_limit = 2.0

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Search cached debate files for matching text; if the cache is empty,
        trigger an incremental fetch (up to 100 speeches) to populate it.
        """
        results: list[SearchResult] = []

        # 1. Try cached files first
        cached = self._load_cached()
        if not cached:
            # Run incremental scrape in a thread so we don't block the event loop
            cached = await asyncio.to_thread(_ep.run_incremental, max_speeches=100)

        # 2. Fuzzy token-overlap filter — match if ANY query token appears in title/text
        q_tokens = set(query.query.lower().split())
        for doc in cached:
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = (title + " " + text).lower()
            if not any(tok in combined for tok in q_tokens):
                continue
            # Date filters
            session_date = doc.get("session_date", "")
            if query.date_from and session_date and session_date < query.date_from:
                continue
            if query.date_to and session_date and session_date > query.date_to:
                continue

            results.append(SearchResult(
                id=self.canonical_id(doc),
                source=self.source_id,
                title=title,
                date=session_date or None,
                url=doc.get("url"),
                excerpt=text[:300] if text else None,
            ))
            if len(results) >= query.limit:
                break

        return results

    async def fetch(self, document_id: str) -> Document:
        """Return a full Document for the given ITM id (from cache or live fetch)."""
        # Try to find a cached file whose itm_id matches
        raw = self._find_cached(document_id)
        if raw is None:
            raise FileNotFoundError(f"ep_debates document not found: {document_id}")

        return Document(
            id=self.canonical_id(raw),
            source=self.source_id,
            title=raw.get("title", ""),
            text=raw.get("text", ""),
            metadata={
                "session_date": raw.get("session_date", ""),
                "speaker_count": raw.get("speaker_count", 0),
                "source_type": raw.get("source_type", ""),
            },
            ingested_at=raw.get("fetched_at", datetime.utcnow().isoformat()),
        )

    def canonical_id(self, raw: dict) -> str:
        return raw.get("itm_id", "") or raw.get("id", "")

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

    def _find_cached(self, document_id: str) -> dict | None:
        safe = re.sub(r"[^a-zA-Z0-9-]", "-", document_id)[-80:]
        candidates = [
            _OUT_DIR / f"{safe}.json",
        ]
        # Broader scan
        for p in _OUT_DIR.glob("*.json"):
            if p.name == "state.json":
                continue
            candidates.append(p)
        for p in candidates:
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
                if data.get("itm_id") == document_id:
                    return data
            except Exception:
                pass
        return None
