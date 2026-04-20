"""
Heading-aware Markdown parser using markdown-it-py.

Splits documents at headings and produces semantically coherent chunks
that preserve the heading hierarchy as heading_path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from kg.parser import BaseParser, ParsedChunk, ParsedDocument


@dataclass
class _Section:
    level: int
    heading: str
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.text_parts).strip()


class MarkdownParser(BaseParser):
    extensions = ("md", "mdx", "markdown")
    mime_types = ("text/markdown",)

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, path: Path) -> ParsedDocument:
        doc = ParsedDocument.from_path(path, "text/markdown")
        raw = path.read_text(errors="replace")
        doc.chunks = list(self._chunk(doc.doc_id, raw))
        return doc

    # ── private ───────────────────────────────────────────────────────────────

    def _chunk(self, doc_id: str, text: str) -> Iterator[ParsedChunk]:
        """Split *text* at ATX headings, then window-chunk each section."""
        try:
            sections = list(self._split_sections(text))
        except Exception:
            # Fallback to plain split
            from kg.parser.text import _split_text

            yield from _split_text(text, doc_id, self.chunk_size, self.chunk_overlap)
            return

        position = 0
        heading_stack: list[str] = []

        for section in sections:
            # Maintain heading breadcrumb
            while heading_stack and _heading_level(heading_stack[-1]) >= section.level:
                heading_stack.pop()
            if section.heading:
                heading_stack.append(section.heading)

            words = section.text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, max(1, len(words)), step):
                chunk_text = " ".join(words[start : start + self.chunk_size])
                if chunk_text.strip():
                    yield ParsedChunk.make(
                        doc_id,
                        position,
                        chunk_text,
                        heading_path=list(heading_stack),
                    )
                    position += 1

    @staticmethod
    def _split_sections(text: str) -> Iterator[_Section]:
        """Split at ATX headings (#, ##, ###, …)."""
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        current = _Section(level=0, heading="")
        prev_end = 0

        for m in heading_re.finditer(text):
            # Flush current section
            body = text[prev_end : m.start()].strip()
            if body:
                current.text_parts.append(body)
            if current.text or current.heading:
                yield current

            current = _Section(level=len(m.group(1)), heading=m.group(2).strip())
            prev_end = m.end()

        # Tail
        tail = text[prev_end:].strip()
        if tail:
            current.text_parts.append(tail)
        if current.text or current.heading:
            yield current


def _heading_level(heading: str) -> int:
    m = re.match(r"^(#{1,6})\s", heading)
    return len(m.group(1)) if m else 0
