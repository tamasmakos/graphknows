"""Base parser protocol and shared data types for the ingestion pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

MAX_CHUNK_TOKENS: int = 512
MIN_CHUNK_TOKENS: int = 64
APPROX_CHARS_PER_TOKEN: int = 4  # rough estimate, no tokenizer dep


@dataclass
class TextSegment:
    """One Markdown-formatted chunk, aligned to heading boundaries.

    Attributes:
        text: Markdown text. Always starts with heading or is body under a heading.
        heading_path: Slash-separated heading breadcrumb, e.g. "## Methods / ### NER".
        char_offset: Byte offset in the original source.
        metadata: Parser-specific extras (language, function_name, lineno, etc.).
    """

    text: str
    heading_path: str
    char_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFile:
    """Output of any BaseParser.parse() call.

    Attributes:
        source_id: Deterministic SHA-256 hex of (source_hint + content).
        title: Human-readable title (filename, first heading, or provided).
        parser_name: Identifies which parser produced this output.
        segments: Ordered list of TextSegment chunks ready for extraction.
    """

    source_id: str
    title: str
    parser_name: str
    segments: list[TextSegment]


class BaseParser(ABC):
    """Abstract parser: converts a source into structured Markdown segments."""

    @abstractmethod
    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        """Parse *text* into Markdown-chunked segments.

        Args:
            text: Raw source text (already fetched/read by the caller).
            source_hint: Original path or URL — used for source_id hashing and metadata.
            title: Optional title override. Parser infers from content if empty.

        Returns:
            ParsedFile with ordered, heading-aligned TextSegment list.
        """

    def _make_source_id(self, source_hint: str, text: str) -> str:
        import hashlib

        return hashlib.sha256(f"{source_hint}::{text[:2000]}".encode()).hexdigest()[:16]

    def _chunk_by_headings(
        self,
        sections: list[tuple[str, str]],  # (heading_path, markdown_body)
        max_tokens: int = MAX_CHUNK_TOKENS,
        min_tokens: int = MIN_CHUNK_TOKENS,
    ) -> list[tuple[str, str]]:
        """Merge small sections, split large ones at paragraphs. Returns (heading_path, text) pairs."""
        result: list[tuple[str, str]] = []
        pending_path = ""
        pending_body = ""

        def _tokens(s: str) -> int:
            return len(s) // APPROX_CHARS_PER_TOKEN

        def _flush() -> None:
            nonlocal pending_path, pending_body
            if pending_body.strip():
                result.append((pending_path, pending_body.strip()))
            pending_path = ""
            pending_body = ""

        for path, body in sections:
            combined = f"{''.join(path.split(' / ')[-1:])}\n\n{body}".strip()
            t_combined = _tokens(combined)
            if t_combined > max_tokens:
                _flush()
                result.extend(self._split_large_section(path, body, max_tokens))
            elif t_combined < min_tokens:
                # Small section — merge with pending
                if _tokens(pending_body) + t_combined > max_tokens:
                    _flush()
                    pending_path = path
                    pending_body = combined
                else:
                    if not pending_path:
                        pending_path = path
                    pending_body = (pending_body + "\n\n" + combined).strip()
            else:
                # Normal size — flush any pending and emit alone
                if pending_body.strip():
                    result.append((pending_path, pending_body.strip()))
                    pending_path = ""
                    pending_body = ""
                result.append((path, combined))

        _flush()
        return result

    def _split_large_section(
        self,
        path: str,
        body: str,
        max_tokens: int,
    ) -> list[tuple[str, str]]:
        """Split a single large section at paragraph boundaries."""
        result: list[tuple[str, str]] = []
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = f"{path.split(' / ')[-1]}\n\n" if path else ""
        heading_text = path.split(" / ")[-1].strip()
        for para in paragraphs:
            buf_text = buf.strip()
            if (len(buf_text) // APPROX_CHARS_PER_TOKEN) + (
                len(para) // APPROX_CHARS_PER_TOKEN
            ) + 2 > max_tokens and len(buf_text) > len(heading_text):
                if buf_text:
                    result.append((path, buf_text))
                buf = f"{heading_text}\n\n{para}"
            else:
                buf = (buf + "\n\n" + para).strip()
        if buf.strip():
            result.append((path, buf.strip()))
        return result
