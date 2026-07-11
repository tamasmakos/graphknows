"""Turn retrieved passages into the memory context handed to the answerer.

Memories are presented **chronologically with a human-readable date and no
retrieval rank or score** — position and score anchoring measurably bias the
answerer toward the top hit, and temporal questions need the timeline visible.
Passages whose text carries no parseable date keep their retrieval order and
are appended after the dated ones.
"""

from __future__ import annotations

import re
from datetime import datetime

from dateparser import parse as _parse_date

from evaluation.common.datamodels import RetrievedPassage

MAX_CONTEXT_CHARS = 24_000

# Date shapes that survive graphknows chunking: "8 May, 2023" (LoCoMo message
# prefixes), "May 8, 2023", "2023/05/01" or ISO (session headers we prefix
# onto each turn at ingest time).
_DATE_RES = (
    re.compile(r"\b\d{1,2} [A-Z][a-z]+,? \d{4}\b"),
    re.compile(r"\b[A-Z][a-z]+ \d{1,2},? \d{4}\b"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
)


def extract_date(text: str) -> datetime | None:
    """First parseable calendar date in *text* (scans a bounded prefix)."""
    head = text[:400]
    for pattern in _DATE_RES:
        match = pattern.search(head)
        if match:
            parsed = _parse_date(match.group(0))
            if parsed is not None:
                return parsed.replace(tzinfo=None)
    return None


def format_memories(
    passages: list[RetrievedPassage], *, max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """Render deduped passages as dated, chronologically ordered memory blocks."""
    seen: set[str] = set()
    dated: list[tuple[datetime, str]] = []
    undated: list[str] = []
    for passage in passages:
        text = passage.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        when = extract_date(text)
        if when is None:
            undated.append(text)
        else:
            dated.append((when, f"({when.strftime('%B %d, %Y')}) {text}"))
    dated.sort(key=lambda pair: pair[0])
    blocks = [block for _, block in dated] + undated
    return "\n\n".join(blocks)[:max_chars]
