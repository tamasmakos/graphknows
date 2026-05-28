# STM-002: Modular Parser Pipeline — Markdown-first, heading-chunked output

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

## Goal
A modular parser pipeline where all parsers emit Markdown with preserved heading hierarchy.
Chunks follow heading boundaries — NEVER split mid-sentence or mid-word.

## Key design decision: Markdown-first, heading-boundary chunking
Every parser converts its source to Markdown. Chunking splits at heading boundaries:
- Level 2 headings (`##`) are the primary split points
- If a section is too large (> MAX_TOKENS), split at paragraph boundaries within it
- If a section is too small, merge with next sibling before splitting
- Result: every chunk is valid Markdown starting with a heading, complete sentences only

MAX_CHUNK_TOKENS = 512 (configurable)
MIN_CHUNK_TOKENS = 64 (merge upward if smaller)

## Files to create (ALL new — no existing files to modify)

### services/graphknows/graphknows/ingestion/__init__.py
Empty.

### services/graphknows/graphknows/ingestion/parsers/__init__.py
Empty.

### services/graphknows/graphknows/ingestion/parsers/base.py
```python
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
            combined = f"{''.join(path.split(' / ')[-1:])}\\n\\n{body}".strip()
            if _tokens(combined) <= max_tokens:
                if _tokens(pending_body) + _tokens(combined) > max_tokens:
                    _flush()
                    pending_path = path
                    pending_body = combined
                else:
                    if not pending_path:
                        pending_path = path
                    pending_body = (pending_body + "\\n\\n" + combined).strip()
            else:
                _flush()
                # Split large section at paragraph boundaries
                paragraphs = [p.strip() for p in body.split("\\n\\n") if p.strip()]
                buf = f"{path.split(' / ')[-1]}\\n\\n" if path else ""
                for para in paragraphs:
                    if _tokens(buf) + _tokens(para) + 2 > max_tokens:
                        if buf.strip():
                            result.append((path, buf.strip()))
                        buf = f"{path.split(' / ')[-1]}\\n\\n{para}"
                    else:
                        buf = (buf + "\\n\\n" + para).strip()
                if buf.strip():
                    result.append((path, buf.strip()))

        _flush()
        return result
```

### services/graphknows/graphknows/ingestion/parsers/text.py
PlainTextParser handles .txt, .md, and raw strings.
- If input looks like Markdown (has `#` headings), parse heading structure directly
- If plain text, treat entire content as one `## Content` section, then paragraph-chunk
- Heading detection: lines starting with `#` to `######`
- Build heading_path by tracking current heading hierarchy (e.g. "## Background / ### Results")
- Call `_chunk_by_headings()` on the extracted sections

```python
"""Plain text and Markdown parser."""
from __future__ import annotations
import hashlib
import re
from .base import BaseParser, ParsedFile, TextSegment, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


class PlainTextParser(BaseParser):
    """Parse plain text or Markdown into heading-aligned Markdown segments.

    Handles .txt and .md files, or raw string input.
    """

    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        # ... implement ...
```

Full implementation: extract headings, build section list, call _chunk_by_headings,
produce TextSegment list with proper heading_path and char_offset.
Infer title from first H1 if present, else from source_hint basename, else "Untitled".

### services/graphknows/graphknows/ingestion/parsers/code.py
CodeParser handles .py files via ast.parse, other code files via line-window heuristics.

For Python files:
- `ast.parse(text)` to extract module docstring, class/function definitions
- Each top-level class/function becomes a `## ClassName` or `## function_name` section
- Body of each becomes a fenced code block in Markdown: ```python\n...\n```
- Docstring (if present) placed as prose before the code block
- metadata: {language: "python", chunk_type: "class"|"function"|"module", name: str, lineno: int}

For non-.py code files (js, ts, sql, sh, yaml, json, etc.):
- Heuristic: split on blank lines between "block-like" structures
- Wrap entire content in fenced code block with language tag
- Heading is filename + section index

```python
"""Code parser: Python AST-aware, fallback heuristic for other languages."""
from __future__ import annotations
import ast
import os
from .base import BaseParser, ParsedFile, TextSegment

class CodeParser(BaseParser):
    """Parse source code files into Markdown-formatted segments."""
    # implement parse() for .py via ast, fallback for others
```

### services/graphknows/graphknows/ingestion/parsers/docx.py
DocxParser uses python-docx.
- `Document(io.BytesIO(text.encode()))` — BUT: text input for docx should be base64 or actual bytes
- ACTUALLY: parse() receives raw text won't work for binary docx
- Solution: add `parse_bytes(data: bytes, ...) -> ParsedFile` as alternative entry point
- `parse(text, ...)` raises ValueError("DocxParser requires bytes — use parse_bytes()")
- Extract paragraphs via `doc.paragraphs`, detect headings via `p.style.name` (Heading 1..6)
- Convert to Markdown: `# H1`, `## H2`, body as plain text paragraphs
- Then call `_chunk_by_headings()`

```python
"""DOCX parser using python-docx. Entry point is parse_bytes() for binary files."""
from __future__ import annotations
from .base import BaseParser, ParsedFile, TextSegment

class DocxParser(BaseParser):
    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        raise ValueError("DocxParser requires binary input — use parse_bytes(data: bytes, ...)")

    def parse_bytes(self, data: bytes, source_hint: str = "", title: str = "") -> ParsedFile:
        # python-docx Document from bytes, build MD, chunk by headings
        ...
```

### services/graphknows/graphknows/ingestion/parsers/registry.py
```python
"""Parser registry: dispatch by file extension or explicit hint."""
from __future__ import annotations
from .base import BaseParser
from .text import PlainTextParser
from .code import CodeParser
from .docx import DocxParser

_EXT_MAP: dict[str, type[BaseParser]] = {
    ".txt": PlainTextParser,
    ".md": PlainTextParser,
    ".py": CodeParser,
    ".js": CodeParser,
    ".ts": CodeParser,
    ".sql": CodeParser,
    ".sh": CodeParser,
    ".yaml": CodeParser,
    ".yml": CodeParser,
    ".json": CodeParser,
    ".docx": DocxParser,
}

def get_parser(path_or_hint: str = "") -> BaseParser:
    """Return the appropriate parser for a path or file extension hint.

    Args:
        path_or_hint: File path, URL, or extension string (e.g. ".py").
                      Defaults to PlainTextParser if unrecognized.
    Returns:
        Instantiated BaseParser subclass.
    """
    import os
    ext = os.path.splitext(path_or_hint)[-1].lower()
    cls = _EXT_MAP.get(ext, PlainTextParser)
    return cls()
```

## Add python-docx to pyproject.toml
In services/graphknows/pyproject.toml, add `"python-docx"` to dependencies.

## Write tests (tests/test_parsers.py)

Test cases (all unit, no live deps):
1. PlainTextParser — plain text (no headings) → 1 segment
2. PlainTextParser — markdown with 3 H2 sections → 3 segments with correct heading_path
3. PlainTextParser — very large section → splits at paragraph boundary, no mid-sentence cuts
4. PlainTextParser — tiny sections merge → merged into fewer chunks
5. CodeParser — simple .py with 2 functions → 2 segments, fenced code blocks in text
6. CodeParser — .py with module docstring + class → 3 segments (module, class body)
7. CodeParser — .js file → 1 segment with ```js``` fenced block
8. DocxParser — parse() raises ValueError
9. DocxParser — parse_bytes() with minimal in-memory docx (use python-docx to build fixture)
10. get_parser(".py") → CodeParser; get_parser(".docx") → DocxParser; get_parser("") → PlainTextParser
11. ParsedFile.segments are ordered (char_offset ascending)
12. No segment has text ending with a partial word (spot-check mid-word split)

## Acceptance criteria
- All 12+ test cases pass
- ruff clean, mypy clean
- `python3 scripts/check_imports.py` clean (python-docx declared)
- No HTTP, no DB calls in parsers — pure text processing
