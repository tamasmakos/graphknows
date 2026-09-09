"""The parser seam: what a parser is, and which one a MIME type reaches.

Parsers are looked up by the source's MIME type, so a pack adds a format by
registering one rather than by editing a dispatch in the ingest path.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from graphknows.models.segment import Segment
from graphknows.models.source import Source


@runtime_checkable
class Parser(Protocol):
    """Cuts one source's bytes into the segments extraction will cite.

    Attributes:
        mimes: The MIME types this parser claims.
        version: Parser identity for provenance; a change re-parses a source.
    """

    mimes: frozenset[str]
    version: str

    def parse(self, source: Source, data: bytes) -> Iterable[Segment]:
        """Yield the segments of *data*, in source order."""
        ...


class ParserRegistry:
    """MIME-keyed parser lookup.

    Registering a parser that claims a MIME another parser already holds
    replaces it: a caller-supplied parser is meant to override the pack's.
    """

    def __init__(self, parsers: Iterable[Parser] = ()) -> None:
        self._by_mime: dict[str, Parser] = {}
        for parser in parsers:
            self.register(parser)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._by_mime)!r})"

    def __iter__(self) -> Iterator[str]:
        """The MIME types currently claimed."""
        return iter(self._by_mime)

    def register(self, parser: Parser) -> None:
        """Claim every MIME type *parser* declares, overriding any holder."""
        if not parser.mimes:
            raise ValueError(f"{parser!r} claims no MIME type, so nothing can reach it")
        for mime in parser.mimes:
            self._by_mime[mime] = parser

    def for_mime(self, mime: str) -> Parser | None:
        """The parser claiming *mime*, or ``None`` when no parser does."""
        return self._by_mime.get(mime)
