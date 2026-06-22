"""Temporal normalization for ingested memory.

Resolves ``DATE``/``TIME`` entity spans (as produced by the spaCy-mined GLiNER2
extractor) into canonical ISO strings with a granularity tag.  Relative
expressions ("last Saturday", "two weeks ago") resolve against an anchor
datetime when one is supplied; absolute expressions ("2022", "25 May 2023")
resolve standalone.  No domain constants — the resolution is fully library
driven (``dateparser``), so it generalises across locales and corpora.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

# dateparser resolves "Saturday", "2 weeks ago", "last week" but not the
# "last/this/past <weekday>" idiom.  Stripping the leading qualifier and
# leaning on PREFER_DATES_FROM=past recovers it without weekday hardcoding.
_LEADING_QUALIFIER = re.compile(r"^(?:the\s+)?(?:last|past|this(?:\s+past)?)\s+", re.IGNORECASE)

# spaCy NER labels that carry temporal meaning. Matched case-insensitively
# against the canonical entity ``type`` assigned by the extractor.
TEMPORAL_TYPES = frozenset({"DATE", "TIME"})

# Map dateparser's detected period to an ISO format that preserves only the
# granularity actually expressed in the source span.
_GRANULARITY_FMT: dict[str, str] = {
    "year": "%Y",
    "month": "%Y-%m",
    "week": "%Y-%m-%d",
    "day": "%Y-%m-%d",
    "time": "%Y-%m-%dT%H:%M",
}


def _anchor_to_datetime(anchor: Any) -> datetime | None:
    """Coerce an anchor (datetime, unix ts, or ISO/unix string) to datetime."""
    if anchor is None or isinstance(anchor, bool):
        return None
    if isinstance(anchor, datetime):
        return anchor
    if isinstance(anchor, (int, float)):
        try:
            return datetime.fromtimestamp(float(anchor))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(anchor, str) and anchor.strip():
        raw = anchor.strip()
        try:
            return datetime.fromtimestamp(float(raw))
        except (OverflowError, OSError, ValueError):
            pass
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def query_date_candidates(query: str) -> list[str]:
    """Extract ISO date candidates from a query for temporal graph matching.

    Returns day-level and year-level ISO strings for every digit-bearing date
    mention.  Word-only matches (dateparser parses "do"/"did" as dates) are
    filtered out to avoid spurious matches.
    """
    from dateparser.search import search_dates

    try:
        found = search_dates(query, settings={"PREFER_DATES_FROM": "past"})
    except Exception as exc:
        log.debug("query date search failed for %r: %s", query, exc)
        return []
    if not found:
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for text, dt in found:
        if not any(ch.isdigit() for ch in text):
            continue
        for iso in (dt.strftime("%Y-%m-%d"), dt.strftime("%Y")):
            if iso not in seen:
                seen.add(iso)
                candidates.append(iso)
    return candidates


def _parse_one(parser: Any, text: str) -> Any:
    """Run one dateparser pass, swallowing locale errors."""
    if not text:
        return None
    try:
        return parser.get_date_data(text)
    except Exception as exc:  # dateparser can raise on odd locale input
        log.debug("temporal parse failed for %r: %s", text, exc)
        return None


def resolve_temporal(
    entities: list[dict[str, Any]],
    anchor: Any = None,
) -> list[dict[str, str]]:
    """Resolve temporal entity spans to canonical ISO dates.

    Args:
        entities: extractor entities, each a dict with ``name`` and ``type``.
        anchor: base datetime/timestamp for relative-date resolution; when
            ``None`` only absolute expressions resolve.

    Returns:
        A de-duplicated list of ``{"raw", "iso", "granularity"}`` dicts.
        Spans that do not resolve to a date are dropped.
    """
    spans = [e for e in entities if str(e.get("type", "")).upper() in TEMPORAL_TYPES]
    if not spans:
        return []

    from dateparser.date import DateDataParser

    base = _anchor_to_datetime(anchor)
    settings: dict[str, Any] = {"PREFER_DATES_FROM": "past"}
    if base is not None:
        settings["RELATIVE_BASE"] = base
    parser = DateDataParser(settings=settings)

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ent in spans:
        raw = str(ent.get("name", "")).strip()
        key = raw.lower()
        if not raw or key in seen:
            continue
        data = _parse_one(parser, raw)
        if data is None or data.date_obj is None:
            stripped = _LEADING_QUALIFIER.sub("", raw, count=1)
            data = _parse_one(parser, stripped) if stripped != raw else None
        if data is None or data.date_obj is None:
            continue
        dt = data.date_obj
        period = data.period or "day"
        iso = dt.strftime(_GRANULARITY_FMT.get(period, "%Y-%m-%d"))
        seen.add(key)
        out.append({"raw": raw, "iso": iso, "granularity": period})
    return out
