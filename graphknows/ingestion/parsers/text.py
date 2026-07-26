"""Plain text and Markdown parser."""

from __future__ import annotations

import os
import re

from .base import BaseParser, ParsedFile

HEADING_RE = re.compile(r"^(#{1,6})\s+([^\n]+)$", re.MULTILINE)


class PlainTextParser(BaseParser):
    """Parse plain text or Markdown into heading-aligned Markdown segments.

    Handles .txt and .md files, or raw string input.
    """

    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        """Parse plain text or Markdown into heading-aligned Markdown segments.

        Args:
            text: Raw text or Markdown content.
            source_hint: Original path or URL for metadata and hashing.
            title: Optional title override.

        Returns:
            ParsedFile with ordered TextSegment list.
        """
        text = text.replace("\r\n", "\n")
        source_id = self._make_source_id(source_hint, text)

        if not title:
            h1_match = re.search(r"^#\s+([^\n]+)$", text, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
            elif source_hint:
                title = os.path.basename(source_hint)
            else:
                title = "Untitled"

        matches = list(HEADING_RE.finditer(text))
        # Plain text is one implicit section; Markdown gets a heading hierarchy.
        sections = (
            self._sections_from_headings(text, matches) if matches else [("## Content", text)]
        )
        chunks = self._chunk_by_headings(sections)

        return ParsedFile(
            source_id=source_id,
            title=title,
            parser_name="PlainTextParser",
            segments=self._segments_from_chunks(chunks, text),
        )

    @staticmethod
    def _sections_from_headings(text: str, matches: list[re.Match[str]]) -> list[tuple[str, str]]:
        """Pair each heading's breadcrumb path with the body that follows it."""
        sections: list[tuple[str, str]] = []
        stack: list[tuple[int, str]] = []
        for i, match in enumerate(matches):
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(0)))

            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(
                (" / ".join(item[1] for item in stack), text[match.end() : body_end].strip())
            )
        return sections
