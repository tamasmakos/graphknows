"""Temporal normalization for ingested memory.

Resolves ``DATE``/``TIME`` entity spans (as produced by the spaCy-mined GLiNER2
extractor) into canonical ISO strings with a granularity tag.  Relative
expressions ("last Saturday", "two weeks ago") resolve against an anchor
datetime when one is supplied; absolute expressions ("2022", "25 May 2023")
resolve standalone.  No domain constants — the resolution is fully library
driven (``dateparser``), so it generalises across locales and corpora.

Bi-temporal helpers (WS-5)
---------------------------
``filter_by_validity_window`` — given a list of relation-edge dicts (each with
optional ``valid_at``, ``invalid_at``, ``created_at`` ISO strings) and an
optional ``as_of`` anchor, returns only those edges whose valid-time window
covers the anchor, boosted by recency.  This is the public surface consumed by
wave-2 workstreams (deterministic rerank + agentic retrieval).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
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


def _iso_to_datetime(iso: str | None) -> datetime | None:
    """Parse an ISO-8601 string to an aware or naive datetime, return None on failure."""
    if not iso:
        return None
    raw = str(iso).strip()
    try:
        dt = datetime.fromisoformat(raw)
        # Normalise to UTC-aware if tzinfo present, else treat as naive.
        return dt
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Bi-temporal window filter (WS-5 public API)
# ---------------------------------------------------------------------------


def filter_by_validity_window(
    edges: list[dict[str, Any]],
    as_of: Any = None,
    *,
    include_open: bool = True,
) -> list[dict[str, Any]]:
    """Filter relation edges by their valid-time window and attach a recency score.

    Public signature for wave-2 workstreams (deterministic rerank + agentic retrieval).

    Each edge dict may carry the following temporal properties (all optional ISO-8601
    strings):

    - ``valid_at``   — the start of the period when the fact holds in the world.
                       Aliases ``valid_from`` for backward compat with STM edges.
    - ``invalid_at`` — the exclusive end of the valid period (``None`` / absent means
                       the fact is still current).  Aliases ``valid_to``.
    - ``created_at`` — transaction time; when the fact was ingested (tie-break for
                       recency scoring).

    Args:
        edges: List of edge property dicts to filter.
        as_of: Optional anchor point for "as of <date>" matching.  Accepts the
               same types as ``_anchor_to_datetime`` (datetime, unix ts, ISO str).
               When ``None`` only open (still-current) edges are retained by default,
               unless ``include_open`` is False.
        include_open: When ``True`` (default) edges with no ``invalid_at`` (i.e.
                      still current) always pass the window check, regardless of
                      whether *as_of* is set.  Set to ``False`` to require an explicit
                      ``valid_at`` value.

    Returns:
        A filtered and sorted list of edge dicts, each augmented with:
        - ``_temporal_score`` (float 0–1): recency signal; higher = more recent.
          Edges with no valid_at receive 0.5 (neutral).  Can be used as a
          multiplicative boost in RRF scoring.
        - ``_is_current`` (bool): True when ``invalid_at`` is absent/None (fact is
          still asserted as of the as_of point or right now).
    """
    anchor_dt: datetime | None = _anchor_to_datetime(as_of)
    now_dt: datetime = datetime.now(timezone.utc)
    # Use now if no anchor specified (matches "current state" queries).
    ref_dt: datetime = anchor_dt if anchor_dt is not None else now_dt

    results: list[dict[str, Any]] = []
    for edge in edges:
        # Resolve property aliases.
        raw_valid_at = edge.get("valid_at") or edge.get("valid_from")
        raw_invalid_at = edge.get("invalid_at") or edge.get("valid_to")

        valid_at_dt = _iso_to_datetime(str(raw_valid_at)) if raw_valid_at else None
        invalid_at_dt = _iso_to_datetime(str(raw_invalid_at)) if raw_invalid_at else None
        is_current = invalid_at_dt is None

        # Make comparison timestamps timezone-consistent.
        def _norm(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        valid_at_dt = _norm(valid_at_dt)
        invalid_at_dt = _norm(invalid_at_dt)
        ref_norm = _norm(ref_dt)

        # Window check — every edge must pass ALL applicable constraints:
        # 1. If valid_at is set, the reference point must be >= valid_at
        #    (the fact must have started by the reference point).
        # 2. If invalid_at is set (tombstoned), the reference point must be
        #    strictly before invalid_at (the fact hasn't been superseded yet).
        # 3. An edge with no valid_at and no invalid_at (undated, still current)
        #    passes when include_open=True (the default for "current state" queries).
        passes = True

        # Constraint 1: has the fact started yet?
        if valid_at_dt is not None and ref_norm is not None and ref_norm < valid_at_dt:
            passes = False  # fact hasn't started yet at the reference point

        # Constraint 2: has the fact been superseded already?
        if invalid_at_dt is not None and ref_norm is not None and ref_norm >= invalid_at_dt:
            passes = False  # fact was tombstoned before the reference point

        if not passes:
            # Skip — either the fact hasn't started yet, or was superseded by ref.
            continue

        if is_current and not include_open and valid_at_dt is None:
            # Caller wants only explicitly time-stamped edges.
            continue

        # Recency score: normalise valid_at relative to a 5-year window.
        # Edges with no valid_at get 0.5 (neutral); edges at or after ref get 1.0.
        _WINDOW_SECS = 5 * 365.25 * 86_400  # 5 years in seconds
        if valid_at_dt is None:
            score = 0.5
        else:
            age_secs = (ref_norm - valid_at_dt).total_seconds() if ref_norm else 0.0
            score = max(0.0, min(1.0, 1.0 - age_secs / _WINDOW_SECS))

        enriched = dict(edge)
        enriched["_temporal_score"] = round(score, 4)
        enriched["_is_current"] = is_current
        results.append(enriched)

    # Drop None sentinels left from tombstoned edges.
    out = [e for e in results if e is not None]
    # Sort: current edges first, then by descending temporal score.
    out.sort(key=lambda e: (not e["_is_current"], -e["_temporal_score"]))
    return out


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
