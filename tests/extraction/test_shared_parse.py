"""One spaCy parse per chunk, shared across the segment.

Three call sites want the same window parsed — lexical label selection,
frame-candidate ranking, and the modality pass. Sharing one model instance
already avoided loading ``en_core_web_lg`` twice; each call still ran the
pipeline again.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import GLiNER2EntityExtractor

pytestmark = pytest.mark.unit


class _CountingPipeline:
    """Stands in for the spaCy pipeline, counting how often it actually runs."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> Any:
        self.calls += 1
        return f"doc::{text}"


def _extractor() -> tuple[GLiNER2EntityExtractor, _CountingPipeline]:
    pipeline = _CountingPipeline()
    miner = MagicMock()
    miner.nlp = pipeline
    return GLiNER2EntityExtractor(schema_miner=miner), pipeline


def test_the_same_text_is_parsed_once() -> None:
    ex, pipeline = _extractor()
    docs = [ex.parse("Melanie moved to Berlin.") for _ in range(3)]
    assert pipeline.calls == 1, f"parsed {pipeline.calls} times, expected 1"
    assert len({id(d) for d in docs}) == 1, "callers got different Doc objects"


def test_a_different_text_is_parsed_again() -> None:
    ex, pipeline = _extractor()
    ex.parse("first window")
    ex.parse("second window")
    assert pipeline.calls == 2


def test_the_memo_holds_one_entry_so_it_cannot_grow() -> None:
    """Bounded by construction: the sharing is within a segment, not across them."""
    ex, pipeline = _extractor()
    ex.parse("a")
    ex.parse("b")
    ex.parse("a")  # evicted by "b" — re-parsed, never stale
    assert pipeline.calls == 3


def test_evicting_never_returns_the_wrong_doc() -> None:
    """The failure that would matter: a caller receiving another chunk's parse."""
    ex, _ = _extractor()
    assert ex.parse("alpha") == "doc::alpha"
    assert ex.parse("beta") == "doc::beta"
    assert ex.parse("alpha") == "doc::alpha"


def test_parse_is_available_before_anything_touches_nlp() -> None:
    """It must not depend on some earlier call having warmed the pipeline."""
    ex, pipeline = _extractor()
    assert ex.parse("cold start") == "doc::cold start"
    assert pipeline.calls == 1
