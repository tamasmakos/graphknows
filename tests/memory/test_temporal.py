"""Unit tests for temporal normalization and query date extraction."""

from __future__ import annotations

from datetime import datetime

import pytest

from graphknows.memory.temporal import query_date_candidates, resolve_temporal

ANCHOR = datetime(2023, 5, 25)


def _ents(*names: str) -> list[dict[str, str]]:
    return [{"name": n, "type": "DATE"} for n in names]


def test_absolute_year_granularity():
    out = resolve_temporal(_ents("2022"))
    assert out == [{"raw": "2022", "iso": "2022", "granularity": "year"}]


def test_absolute_day():
    out = resolve_temporal(_ents("25 May 2023"), ANCHOR)
    assert {"raw": "25 May 2023", "iso": "2023-05-25", "granularity": "day"} in out


def test_relative_offset_resolves_against_anchor():
    out = {o["raw"]: o["iso"] for o in resolve_temporal(_ents("two weeks ago"), ANCHOR)}
    assert out["two weeks ago"] == "2023-05-11"


def test_last_weekday_fallback():
    # "last Saturday" is not parsed by dateparser directly; the qualifier-strip
    # fallback recovers it against the anchor (Thu 25 May 2023 -> 20 May).
    out = {o["raw"]: o["iso"] for o in resolve_temporal(_ents("last Saturday"), ANCHOR)}
    assert out["last Saturday"] == "2023-05-20"


def test_non_temporal_entities_ignored():
    assert resolve_temporal([{"name": "Melanie", "type": "PERSON"}]) == []


def test_unresolvable_span_dropped():
    assert resolve_temporal(_ents("sometime maybe")) == []


def test_dedup_by_surface_form():
    out = resolve_temporal(_ents("2022", "2022"))
    assert len(out) == 1


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What happened on 19 May 2023?", "2023-05-19"),
        ("events in 2022", "2022"),
    ],
)
def test_query_date_candidates_extracts_digit_dates(query, expected):
    assert expected in query_date_candidates(query)


def test_query_date_candidates_filters_word_noise():
    # "When did the race happen?" — dateparser parses "did" as a date; the
    # digit filter must drop it.
    assert query_date_candidates("When did the race happen?") == []
