"""Turn retrieved passages into the memory context handed to the answerer.

Memories are presented **chronologically with a human-readable date and no
retrieval rank or score** — position and score anchoring measurably bias the
answerer toward the top hit, and temporal questions need the timeline visible.
Passages whose text carries no parseable date keep their retrieval order and
are appended after the dated ones.
"""

from __future__ import annotations

from datetime import datetime

from evaluation.common.datamodels import RetrievedPassage
from graphknows import MAX_CONTEXT_CHARS, date_in

# The date shapes and the character budget are the library's, imported rather
# than restated: they were duplicated here verbatim — same three regexes, same
# 72k — and a duplicated constant is one that drifts. What stays local is the
# one thing that genuinely differs: these passages arrive from the MCP wire
# without a ``ts`` field, so their date has to be scraped back out of the text
# and PREFIXED, where a typed Hit carries its own and renders it directly.
__all__ = ["MAX_CONTEXT_CHARS", "extract_date", "format_memories"]


def extract_date(text: str) -> datetime | None:
    """First parseable calendar date in *text* (scans a bounded prefix)."""
    return date_in(text)


def format_memories(passages: list[RetrievedPassage], *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render deduped passages as dated, chronologically ordered memory blocks.

    *passages* arrive in RELEVANCE order (the retriever's ranking). Selection
    happens in that order and drops WHOLE passages once the budget is spent;
    only the surviving set is then sorted chronologically for display, because
    the timeline needs to read oldest-first while the choice of what to keep
    must follow relevance.

    Previously this sorted chronologically FIRST and truncated the joined string
    (``"\\n\\n".join(blocks)[:max_chars]``), which silently deleted the newest
    memories mid-sentence: on a May-September conversation the generator never
    saw anything after early June regardless of how many passages it was given.
    That produced "the memories do not mention X" for facts that had been
    retrieved correctly, and it made raising the passage count actively harmful
    — more old passages consumed the same budget and pushed recent evidence
    further out of the window.
    """
    seen: set[str] = set()
    kept: list[str] = []
    used = 0
    sep = 2  # the "\n\n" join separator
    for passage in passages:
        text = passage.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        when = extract_date(text)
        block = f"({when.strftime('%B %d, %Y')}) {text}" if when else text
        cost = len(block) + (sep if kept else 0)
        if used + cost > max_chars:
            # Skip this one but keep scanning: a shorter, lower-ranked passage
            # may still fit, and a partial block helps nobody.
            continue
        used += cost
        kept.append(block)

    dated: list[tuple[datetime, str]] = []
    undated: list[str] = []
    for block in kept:
        when = extract_date(block)
        (dated.append((when, block)) if when else undated.append(block))
    dated.sort(key=lambda pair: pair[0])
    return "\n\n".join([b for _, b in dated] + undated)
