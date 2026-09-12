"""The ingest path's parser: plain text and Markdown into heading-aligned segments.

Holds the shared segment types and text helpers too. They used to sit in a
``base`` module behind a ``BaseParser`` ABC — but the ABC had one implementation
and the inheritance ran backwards: every line of chunking logic lived in the
"abstract" base while the concrete class contributed only the heading split.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any

MAX_CHUNK_TOKENS: int = 384
MIN_CHUNK_TOKENS: int = 48
MAX_CHUNK_OVERLAP: int = 64  # trailing tokens carried into the next chunk
APPROX_CHARS_PER_TOKEN: int = 4  # rough estimate, no tokenizer dep

HEADING_RE = re.compile(r"^(#{1,6})\s+([^\n]+)$", re.MULTILINE)

_CONTENT_HEADING_RE = re.compile(r"^##\s*Content\s*\n+", re.IGNORECASE)


def strip_content_heading(text: str) -> str:
    """Remove the synthetic ``## Content`` heading PlainTextParser adds.

    Plain text has no headings, so the parser invents one to give the chunk a
    heading path. It is an artifact of chunking, not of the source, so both the
    ingest path (before embedding) and the recall path (before display) strip
    it — they each had their own copy of this regex.
    """
    return _CONTENT_HEADING_RE.sub("", text).strip()


def search_surface(text: str, speaker: str = "") -> str:
    """The text as it should be EMBEDDED and lexically scanned.

    An utterance usually omits its own speaker: "I'm opening a dance studio" is
    a fact about Gina and names her nowhere. Stored verbatim, no query asking
    about Gina can reach it — not lexically, and not by embedding either. So the
    retrieval surface carries the attribution while ``CHUNK.text`` stays exactly
    what was said, and the EXTRACTOR keeps seeing the bare text (putting the
    name in its input is what mints speakers as ENTITY nodes).

    Returns ``text`` unchanged when there is no speaker, or when the text
    already opens with it — a caller that flattens "Gina: ..." into the content
    itself must not end up with "Gina: Gina: ...".
    """
    name = (speaker or "").strip()
    body = (text or "").strip()
    if not name or not body or body.casefold().startswith(f"{name.casefold()}:"):
        return body
    return f"{name}: {body}"


def _tokens(s: str) -> int:
    return len(s) // APPROX_CHARS_PER_TOKEN


class _PendingMerge:
    """Accumulator for sections too small to stand alone as a chunk.

    Small sections merge together until one more would blow the budget. Holding
    the buffer here instead of in closure state is what keeps
    ``_chunk_by_headings`` a flat three-way dispatch.
    """

    def __init__(self, max_tokens: int) -> None:
        self._max = max_tokens
        self._path = ""
        self._body = ""

    def flush(self) -> list[tuple[str, str]]:
        """Emit whatever has accumulated (possibly nothing) and reset."""
        out = [(self._path, self._body.strip())] if self._body.strip() else []
        self._path = ""
        self._body = ""
        return out

    def absorb(self, path: str, combined: str) -> list[tuple[str, str]]:
        """Merge a small section in, flushing first if it would overflow."""
        if _tokens(self._body) + _tokens(combined) > self._max:
            out = self.flush()
            self._path, self._body = path, combined
            return out
        if not self._path:
            self._path = path
        self._body = (self._body + "\n\n" + combined).strip()
        return []


def _tail_units(seq: list[str], overlap: int) -> tuple[list[str], int]:
    """Trailing whole units of *seq* fitting in the overlap budget (order kept)."""
    carried: list[str] = []
    toks = 0
    for u in reversed(seq):
        ut = max(1, _tokens(u))
        if toks + ut > overlap:
            break
        carried.insert(0, u)
        toks += ut
    return carried, toks


def _pack_sentences(unit: str, max_tokens: int, overlap: int) -> list[str]:
    """Sentence-split one over-long unit so no piece exceeds the budget."""
    from graphknows.ingestion.parsers._sentences import split_sentences

    out: list[str] = []
    sub: list[str] = []
    sub_tokens = 0
    for sent in split_sentences(unit):
        if sub and sub_tokens + max(1, _tokens(sent)) > max_tokens:
            out.append(" ".join(sub))
            sub, sub_tokens = _tail_units(sub, overlap)
        sub.append(sent)
        sub_tokens += max(1, _tokens(sent))
    if sub:
        out.append(" ".join(sub))
    return out


def _pack_units(units: list[str], max_tokens: int, overlap: int) -> list[str]:
    """Pack sentences into ``<= max_tokens`` bodies, with overlap."""
    bodies: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for unit in units:
        # A single unit over budget (a very long prose run): flush the buffer,
        # then sentence-split the unit so no chunk exceeds the budget.
        if max(1, _tokens(unit)) > max_tokens:
            if buf:
                bodies.append(" ".join(buf))
                buf, buf_tokens = [], 0
            bodies.extend(_pack_sentences(unit, max_tokens, overlap))
            continue
        if buf and buf_tokens + max(1, _tokens(unit)) > max_tokens:
            bodies.append(" ".join(buf))
            buf, buf_tokens = _tail_units(buf, overlap)
        buf.append(unit)
        buf_tokens += max(1, _tokens(unit))
    if buf:
        bodies.append(" ".join(buf))
    return bodies


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
    """Output of a ``PlainTextParser.parse()`` call.

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


class PlainTextParser:
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

    def _make_source_id(self, source_hint: str, text: str) -> str:
        return hashlib.sha256(f"{source_hint}::{text[:2000]}".encode()).hexdigest()[:16]

    def _chunk_by_headings(
        self,
        sections: list[tuple[str, str]],  # (heading_path, markdown_body)
        max_tokens: int = MAX_CHUNK_TOKENS,
        min_tokens: int = MIN_CHUNK_TOKENS,
    ) -> list[tuple[str, str]]:
        """Merge small sections, split large ones at paragraphs. Returns (heading_path, text) pairs."""
        result: list[tuple[str, str]] = []
        pending = _PendingMerge(max_tokens)

        for path, body in sections:
            combined = f"{''.join(path.split(' / ')[-1:])}\n\n{body}".strip()
            size = _tokens(combined)
            if size > max_tokens:
                result.extend(pending.flush())
                result.extend(self._split_large_section(path, body, max_tokens))
            elif size < min_tokens:
                result.extend(pending.absorb(path, combined))
            else:
                result.extend(pending.flush())
                result.append((path, combined))

        result.extend(pending.flush())
        return result

    def _segments_from_chunks(self, chunks: list[tuple[str, str]], text: str) -> list[TextSegment]:
        """Turn ``(heading_path, chunk_text)`` pairs into offset-anchored segments.

        This loop was written out three times (twice inside one method) and
        drifted between copies.
        """
        return [
            TextSegment(
                text=chunk_text,
                heading_path=path,
                char_offset=self._find_offset(text, chunk_text),
            )
            for path, chunk_text in chunks
        ]

    def _find_offset(self, text: str, chunk_text: str) -> int:
        """Offset of the first identifiable line of *chunk_text* within *text*."""
        for raw in chunk_text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            idx = text.find(line)
            if idx != -1:
                return idx
        return 0

    def _split_large_section(
        self,
        path: str,
        body: str,
        max_tokens: int,
        overlap: int = MAX_CHUNK_OVERLAP,
    ) -> list[tuple[str, str]]:
        """Split a large section to ``<= max_tokens``, never breaking a word.

        Sentences are the unit, so a chunk boundary always falls where a sentence
        ends. Consecutive chunks overlap by up to ``overlap`` tokens: when a chunk
        is flushed, its trailing whole sentences that fit in the overlap budget
        seed the next chunk, so a fact spanning a chunk boundary is not lost to
        retrieval.
        """
        from graphknows.ingestion.parsers._sentences import split_sentences

        heading_text = path.split(" / ")[-1].strip()
        bodies = _pack_units(split_sentences(body), max_tokens, overlap)

        return [
            (path, f"{heading_text}\n\n{b}" if heading_text else b) for b in bodies if b.strip()
        ]
