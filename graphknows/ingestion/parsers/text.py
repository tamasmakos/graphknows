"""Plain text and Markdown parser."""

from __future__ import annotations

import os
import re

from .base import BaseParser, ParsedFile, TextSegment

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
        if not matches:
            # Plain text - treat as one section and paragraph-chunk if needed
            sections = [("## Content", text)]
            chunks = self._chunk_by_headings(sections)
            segments = []
            for path, chunk_text in chunks:
                offset = self._find_offset(text, chunk_text)
                segments.append(
                    TextSegment(
                        text=chunk_text,
                        heading_path=path,
                        char_offset=offset,
                    )
                )
        else:
            # Markdown with headings - build heading hierarchy
            sections = []
            stack: list[tuple[int, str]] = []
            for i, match in enumerate(matches):
                level = len(match.group(1))
                heading_line = match.group(0)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading_line))
                path = " / ".join(item[1] for item in stack)

                body_start = match.end()
                if i + 1 < len(matches):
                    body_end = matches[i + 1].start()
                else:
                    body_end = len(text)
                body = text[body_start:body_end].strip()
                sections.append((path, body))

            chunks = self._chunk_by_headings(sections)
            segments = []
            for path, chunk_text in chunks:
                offset = self._find_offset(text, chunk_text)
                segments.append(
                    TextSegment(
                        text=chunk_text,
                        heading_path=path,
                        char_offset=offset,
                    )
                )

        return ParsedFile(
            source_id=source_id,
            title=title,
            parser_name="PlainTextParser",
            segments=segments,
        )

    def _find_offset(self, text: str, chunk_text: str) -> int:
        """Find the byte offset of the first identifiable line in chunk_text."""
        for line in chunk_text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                idx = text.find(line)
                if idx != -1:
                    return idx
            if line.startswith("#"):
                idx = text.find(line)
                if idx != -1:
                    return idx
        return 0
