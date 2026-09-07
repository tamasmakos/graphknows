"""Temporal normalization for ingested memory.

Resolves ``DATE``/``TIME`` entity spans (as produced by the spaCy-mined GLiNER2
extractor) into canonical ISO strings with a granularity tag.  Relative
expressions ("last Saturday", "two weeks ago") resolve against an anchor
datetime when one is supplied; absolute expressions ("2022", "25 May 2023")
resolve standalone.  No domain constants — the resolution is fully library
driven (``dateparser``), so it generalises across locales and corpora.

A leaf utility, not a channel: it implements no ``Channel`` contract and imports
nothing from graphknows, but both pipelines call it — ingestion to write
``TEMPORAL`` nodes, retrieval to gate and anchor date-shaped questions — so it
sits on the shared bottom rung alongside ``models`` and ``settings``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import lru_cache
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
    """Coerce an anchor (datetime, unix ts, ISO string, or prose date) to datetime.

    Prose matters: a conversation's timestamp is written the way people write
    them ("4:04 pm on 20 January, 2023", "Jan 20 2023"), not as ISO. Accepting
    only unix/ISO made every such anchor return None, which does not fail — it
    silently drops RELATIVE_BASE, so the turn's relative dates resolve against
    *ingestion* time instead. Measured on LoCoMo conv-30: 42 of 62 TEMPORAL
    nodes dated 2025/2026 for a conversation that ran January-July 2023.
    """
    if anchor is None or isinstance(anchor, bool):
        return None
    if isinstance(anchor, datetime):
        return anchor
    if isinstance(anchor, int | float):
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
            pass
        # Prose fallback. dateparser is already this module's date engine, so
        # the anchor is parsed by the same rules as the spans it anchors.
        from dateparser import parse as _parse  # type: ignore[import-untyped]

        try:
            return _parse(raw)  # type: ignore[no-any-return]
        except Exception as exc:  # dateparser raises on some locale inputs
            log.debug("anchor %r did not parse: %s", raw, exc)
            return None
    return None


# Interrogatives that mean "the answer I want is a date/duration". Used to gate
# the entity-keyed fallback in temporal_traversal: without it that fallback ran
# on every query that merely LACKED a date, which is nearly all of them.
# Measured coverage of the LoCoMo temporal category: 26/26 on conv-30, 275/321
# across all ten conversations. A miss costs the fallback, not the query — the
# other four channels still run.
_ASKS_FOR_A_DATE = re.compile(
    r"\b(when|what date|which date|what year|which year|what month|which month"
    r"|what time|how long|how old|how often|how many (?:days|weeks|months|years))\b",
    re.IGNORECASE,
)


def asks_for_a_date(query: str) -> bool:
    """Whether *query* is asking for a date, duration or frequency as its ANSWER.

    Distinct from :func:`query_date_candidates`, which asks whether the query
    STATES a date. "When did Gina lose her job" states none and asks for one;
    "what happened on 1 February 2023" states one and asks for something else.
    """
    return bool(_ASKS_FOR_A_DATE.search(query or ""))


def query_date_candidates(query: str) -> list[str]:
    """Extract ISO date candidates from a query for temporal graph matching.

    Returns day-level and year-level ISO strings for every digit-bearing date
    mention.  Word-only matches (dateparser parses "do"/"did" as dates) are
    filtered out to avoid spurious matches.
    """
    from dateparser.search import search_dates  # type: ignore[import-untyped]

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


@lru_cache(maxsize=128)
def _parser_for(base: datetime | None) -> Any:
    """The ``DateDataParser`` for one anchor, built once and reused.

    Every chunk in a document shares the same anchor (the session/turn
    timestamp), so building a fresh parser per chunk rebuilt the identical
    thing on every one of them. ``base`` is the whole cache key: ``datetime``
    is hashable, and a bounded cache keeps a long-running process from
    accumulating one parser per anchor it has ever seen.
    """
    from dateparser.date import DateDataParser  # type: ignore[import-untyped]

    settings: dict[str, Any] = {"PREFER_DATES_FROM": "past"}
    if base is not None:
        settings["RELATIVE_BASE"] = base
    return DateDataParser(settings=settings)


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

    base = _anchor_to_datetime(anchor)
    parser = _parser_for(base)

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
