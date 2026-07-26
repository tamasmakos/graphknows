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

# Character budget for the rendered memory block. ~4 chars/token, so 72k chars is
# ~18k tokens — roughly 14% of llama-3.3-70b's 128k window, and enough to hold all
# 50 retrieved LoCoMo passages (~1.3k chars each) without dropping any. The old
# 24k ceiling capped the generator at ~17 passages no matter what gen_context_k
# said, and measured prompts sat at a flat ~7.8k tokens — the cap, not the content,
# was setting the context size.
MAX_CONTEXT_CHARS = 72_000

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
