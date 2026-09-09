"""The parser seam: MIME lookup, override, and protocol conformance."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from graphknows.ingestion.parsers.registry import Parser, ParserRegistry
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source


class _WholeFileParser:
    """A parser emitting the whole source as one segment."""

    def __init__(self, mimes: Iterable[str], version: str = "1") -> None:
        self.mimes = frozenset(mimes)
        self.version = version

    def parse(self, source: Source, data: bytes) -> Iterable[Segment]:
        yield Segment(
            source_id=source.id,
            text=data.decode(),
            kind=SegmentKind.prose,
            byte_range=(0, len(data)),
        )


def test_a_parser_is_reached_by_every_mime_it_claims():
    parser = _WholeFileParser(["text/plain", "text/markdown"])
    registry = ParserRegistry([parser])

    assert registry.for_mime("text/plain") is parser
    assert registry.for_mime("text/markdown") is parser
    assert sorted(registry) == ["text/markdown", "text/plain"]


def test_an_unclaimed_mime_reaches_no_parser():
    assert ParserRegistry().for_mime("application/pdf") is None


def test_a_later_parser_overrides_the_mime_it_shares():
    first = _WholeFileParser(["text/plain"])
    second = _WholeFileParser(["text/plain"])

    registry = ParserRegistry([first])
    registry.register(second)

    assert registry.for_mime("text/plain") is second


def test_a_parser_claiming_no_mime_is_refused():
    with pytest.raises(ValueError, match="no MIME type"):
        ParserRegistry([_WholeFileParser([])])


def test_the_registered_parser_parses_bytes_into_segments():
    source = Source(uri="notes.txt", content_hash="abc", mime="text/plain")
    registry = ParserRegistry([_WholeFileParser(["text/plain"])])

    parser = registry.for_mime(source.mime)
    assert parser is not None
    segments = list(parser.parse(source, b"Hello world."))

    assert [s.text for s in segments] == ["Hello world."]
    assert segments[0].source_id == source.id
    assert isinstance(parser, Parser)
