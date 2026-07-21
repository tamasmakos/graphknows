"""DOCX parser using python-docx. Entry point is parse_bytes() for binary files."""

from __future__ import annotations

import io
import os

from .base import BaseParser, ParsedFile


class DocxParser(BaseParser):
    """Parse .docx files into Markdown-formatted, heading-aligned segments."""

    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        """Reject text parsing because DOCX inputs must be parsed from bytes."""
        raise NotImplementedError(
            "DocxParser requires binary input — use parse_bytes(data: bytes, ...)"
        )

    def parse_bytes(self, data: bytes, source_hint: str = "", title: str = "") -> ParsedFile:
        """Parse binary DOCX data into Markdown-chunked segments.

        Args:
            data: Raw .docx file bytes.
            source_hint: Original path or URL for hashing and metadata.
            title: Optional title override.

        Returns:
            ParsedFile with ordered TextSegment list.
        """
        try:
            from docx import Document
        except ModuleNotFoundError as exc:
            from graphknows.exceptions import MissingExtraError

            raise MissingExtraError("DOCX parsing", "parsers") from exc

        text_parts = self._markdown_lines(Document(io.BytesIO(data)))
        full_text = "\n\n".join(text_parts)
        source_id = self._make_source_id(source_hint, full_text)
        if not title:
            title = os.path.basename(source_hint) if source_hint else "Untitled"

        sections = self._build_docx_sections(text_parts) or [("## Content", full_text)]
        chunks = self._chunk_by_headings(sections)

        return ParsedFile(
            source_id=source_id,
            title=title,
            parser_name="DocxParser",
            segments=self._segments_from_chunks(chunks, full_text),
        )

    @staticmethod
    def _markdown_lines(doc: object) -> list[str]:
        """Flatten paragraphs to text, rendering Word heading styles as ``#`` prefixes."""
        lines: list[str] = []
        for para in doc.paragraphs:  # type: ignore[attr-defined]
            body = para.text.strip()
            style_name = para.style.name if para.style else ""
            if style_name.startswith("Heading "):
                try:
                    level = int(style_name.split()[-1])
                except (ValueError, IndexError):
                    level = 1
                lines.append(f"{'#' * level} {body}")
            elif body:
                lines.append(body)
        return lines

    def _build_docx_sections(self, text_parts: list[str]) -> list[tuple[str, str]]:
        """Convert extracted DOCX text parts into (heading_path, body) sections."""
        sections: list[tuple[str, str]] = []
        current_path = "## Content"
        current_body_parts: list[str] = []
        for line in text_parts:
            if line.startswith("#"):
                if current_body_parts:
                    sections.append((current_path, "\n\n".join(current_body_parts)))
                    current_body_parts = []
                level = len(line) - len(line.lstrip("#"))
                heading = f"{'#' * level} {line.lstrip('#').strip()}"
                current_path = heading
            else:
                current_body_parts.append(line)
        if current_body_parts:
            sections.append((current_path, "\n\n".join(current_body_parts)))
        return sections
