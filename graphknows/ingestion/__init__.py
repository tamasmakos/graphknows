"""Ingestion — the write path: parse -> extract -> pipeline.

Exposes the parser's output types. The Source -> Segment orchestration lives in
:mod:`graphknows.ingestion.pipeline`, which the ``Memory`` facade builds.
"""

from __future__ import annotations

from graphknows.ingestion.parsers.text import ParsedFile, PlainTextParser, TextSegment

__all__ = [
    "ParsedFile",
    "PlainTextParser",
    "TextSegment",
]
