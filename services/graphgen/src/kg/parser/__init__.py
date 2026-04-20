"""
Document parser abstraction for GraphKnows.

Each parser converts a raw file to a list of ParsedChunk objects.
Auto-registration: subclass BaseParser and it becomes available.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type["BaseParser"]] = {}


@dataclass
class ParsedChunk:
    """A single chunk extracted from a document."""

    chunk_id: str
    doc_id: str
    position: int
    text: str
    heading_path: list[str] = field(default_factory=list)
    token_count: int = 0

    @classmethod
    def make(
        cls,
        doc_id: str,
        position: int,
        text: str,
        heading_path: list[str] | None = None,
    ) -> "ParsedChunk":
        chunk_id = f"{doc_id}:{position}"
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            position=position,
            text=text.strip(),
            heading_path=heading_path or [],
            token_count=len(text.split()),
        )


@dataclass
class ParsedDocument:
    """Result of parsing a single file."""

    doc_id: str
    title: str
    source_path: str
    mime_type: str
    content_hash: str
    chunks: list[ParsedChunk] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path, mime_type: str) -> "ParsedDocument":
        raw_bytes = path.read_bytes()
        doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]
        return cls(
            doc_id=doc_id,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        )


class BaseParser(ABC):
    """Base class for all document parsers."""

    #: Supported MIME types — override in subclasses.
    mime_types: tuple[str, ...] = ()
    #: File extensions (without dot) — override in subclasses.
    extensions: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for ext in cls.extensions:
            _REGISTRY[ext.lower()] = cls

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse *path* and return a ParsedDocument with populated chunks."""

    def iter_chunks(
        self,
        path: Path,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ) -> Iterator[ParsedChunk]:
        doc = self.parse(path)
        yield from doc.chunks


# ── Registry helpers ──────────────────────────────────────────────────────────


def get_parser(path: Path) -> BaseParser:
    """Return the appropriate parser for *path* based on its extension."""
    ext = path.suffix.lstrip(".").lower()
    cls = _REGISTRY.get(ext)
    if cls is None:
        # Fallback to plain text
        from kg.parser.text import TextParser

        return TextParser()
    return cls()


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY.keys())
