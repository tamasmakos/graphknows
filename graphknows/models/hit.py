"""One recalled memory, and the one way a ranked list of them renders.

Every adapter over :meth:`graphknows.Memory.recall_memory` — the LangGraph
nodes, the langgraph store, the MCP tool, an evaluation harness — needs the
same three things from a hit: its text, who said it, and when. Before this
module each of them re-derived that projection from an untyped ``dict``, and
they disagreed: the MCP shape carried no ``speaker`` and no ``ts`` at all, so
every consumer downstream of it recovered the date by regex-scraping the
passage text and the attribution not at all.

:class:`Hit` is that shape, once. :func:`render_memories` is the other half —
turning ranked hits into the dated, attributed, chronological block an answerer
reads. Rendering lives here rather than in each caller because the decisions in
it are not presentation taste; they are measured:

- **Selection runs in relevance order, display in chronological order.** Sorting
  first and truncating the joined string deletes the newest memories
  mid-sentence, so a May-September conversation answers "not mentioned" for
  September facts that were retrieved correctly.
- **Whole memories are dropped, never truncated**, and a skipped memory does not
  stop the scan — a shorter, lower-ranked one may still fit.
- **Undated memories keep their retrieval order** and follow the timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import Any

# dateparser ships no py.typed marker and its stubs are a separate package that
# is not a declared dependency — same ignore, same reason, as every other
# dateparser import in this package (see graphknows/channels/temporal.py).
from dateparser import parse as _parse_date  # type: ignore[import-untyped]

# ~4 chars/token, so 72k chars is ~18k tokens — roughly 14% of a 128k window,
# and enough to hold every passage of a large recall without dropping one. The
# older 24k ceiling capped the answerer at ~17 passages no matter how many it
# was given: the cap, not the content, was setting the context size.
MAX_CONTEXT_CHARS = 72_000

# Date shapes that survive chunking, for hits whose date is written into the
# text rather than carried in ``ts``: "8 May, 2023", "May 8, 2023", ISO-ish.
_DATE_PATTERNS = (
    re.compile(r"\b\d{1,2} [A-Z][a-z]+,? \d{4}\b"),
    re.compile(r"\b[A-Z][a-z]+ \d{1,2},? \d{4}\b"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
)


def _naive(when: datetime | None) -> datetime | None:
    """Drop the tzinfo. Mixed aware/naive datetimes raise on comparison."""
    return when.replace(tzinfo=None) if when is not None else None


def date_in(text: str) -> datetime | None:
    """The first calendar date written into *text* (scans a bounded prefix)."""
    head = text[:400]
    for pattern in _DATE_PATTERNS:
        if (match := pattern.search(head)) and (parsed := _parse_date(match.group(0))):
            return _naive(parsed)
    return None


@dataclass
class Hit:
    """One ranked memory returned by recall.

    ``ts`` is deliberately the raw store value ("4:04 pm on 20 January, 2023"),
    not a parsed datetime: it is whatever the turn was timestamped with, and
    normalising it at the store would lose the cases dateparser handles better
    than a format string. :attr:`when` is the parsed view.
    """

    text: str
    speaker: str = ""
    ts: str = ""
    score: float = 0.0
    sources: str = ""
    session_id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    entities: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def stamped(self) -> datetime | None:
        """This hit's own timestamp, parsed. ``None`` when it carries none."""
        return _naive(_parse_date(self.ts)) if self.ts.strip() else None

    @cached_property
    def when(self) -> datetime | None:
        """Best available date: this hit's ``ts``, else a date found in the text."""
        return self.stamped or date_in(self.text)

    def render(self) -> str:
        """This hit as one dated, attributed memory block.

        A hit with no parseable ``ts`` is passed through untouched: its text
        already carries whatever date and speaker it has inline, so prefixing
        would double the date, and dropping it would leave the memory undated
        against a prompt that promises dates.
        """
        text = self.text.strip()
        if self.stamped is None or not text:
            return text
        lead = f"({self.stamped.strftime('%B %d, %Y')})"
        who = self.speaker.strip()
        return f"{lead} {who}: {text}" if who else f"{lead} {text}"

    def to_dict(self) -> dict[str, Any]:
        """The wire shape, for transports that cannot carry the type itself."""
        return {
            "text": self.text,
            "speaker": self.speaker,
            "ts": self.ts,
            "score": self.score,
            "sources": self.sources,
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entities": self.entities,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Hit:
        """Rebuild a hit from its wire shape (the inverse of :meth:`to_dict`)."""
        return cls(
            text=str(raw.get("text", "") or ""),
            speaker=str(raw.get("speaker", "") or ""),
            ts=str(raw.get("ts", "") or ""),
            score=float(raw.get("score", 0.0) or 0.0),
            sources=str(raw.get("sources", "") or ""),
            session_id=str(raw.get("session_id", "") or ""),
            chunk_id=str(raw.get("chunk_id", "") or ""),
            doc_id=str(raw.get("doc_id", "") or ""),
            entities=list(raw.get("entities", []) or []),
            metadata=dict(raw.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class RenderedMemories:
    """The memory block handed to an answerer, plus what it is made of.

    The counts are not decoration. A recall silently falling back to the
    undated branch — because ``ts`` stopped arriving, or stopped parsing —
    reads downstream as a mysteriously worse answer with nothing to point at.
    """

    text: str
    dated: int
    undated: int

    def __str__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text)

    @property
    def shown(self) -> int:
        """How many memories reached the block after dedup and budgeting."""
        return self.dated + self.undated


def render_memories(hits: list[Hit], *, max_chars: int = MAX_CONTEXT_CHARS) -> RenderedMemories:
    """Render ranked *hits* as the deduped, dated, chronological memory block.

    *hits* arrive in relevance order. Selection follows that order and drops
    whole memories once the budget is spent; only the survivors are then sorted
    chronologically, because the timeline reads oldest-first while the choice of
    what to keep must follow the ranking.
    """
    seen: set[str] = set()
    dated: list[tuple[datetime, str]] = []
    undated: list[str] = []
    used = 0
    for hit in hits:
        block = hit.render()
        if not block or block in seen:
            continue
        seen.add(block)
        cost = len(block) + (2 if used else 0)  # the "\n\n" join separator
        if used + cost > max_chars:
            continue  # a shorter, lower-ranked memory may still fit
        used += cost
        if (when := hit.when) is not None:
            dated.append((when, block))
        else:
            undated.append(block)
    dated.sort(key=lambda pair: pair[0])
    return RenderedMemories(
        text="\n\n".join([b for _, b in dated] + undated),
        dated=len(dated),
        undated=len(undated),
    )


__all__ = ["MAX_CONTEXT_CHARS", "Hit", "RenderedMemories", "date_in", "render_memories"]
