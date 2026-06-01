"""Ingestion package — file parsers and parser registry.

Example:
    from graphknows.ingestion import get_parser, ParsedFile
"""

from graphknows.ingestion.parsers.base import BaseParser, ParsedFile, TextSegment
from graphknows.ingestion.parsers.registry import get_parser

__all__ = [
    "BaseParser",
    "ParsedFile",
    "TextSegment",
    "get_parser",
]
