"""Text/date helpers shared by the benchmark dataset modules.

Used by ``evaluation/{beam,locomo,longmem}/dataset.py`` when normalizing raw
benchmark records into :class:`~evaluation.common.datamodels.EvalDocument`.
"""

from __future__ import annotations

from typing import Any

from dateparser import parse as _parse_date

from evaluation.common.datamodels import EvalDocument


def clip(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars``, appending an ellipsis when cut."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def reference_date(documents: list[EvalDocument]) -> str:
    """Human-readable date of the newest document (anchors relative time)."""
    parsed = [dt for doc in documents if doc.anchor_date and (dt := _parse_date(doc.anchor_date))]
    return max(parsed).strftime("%d %B %Y") if parsed else ""


def message_turns(messages: list[Any], *, timestamp: str = "") -> list[dict[str, str]]:
    """Chat messages as ingestable turns, structure intact.

    ``role`` and ``timestamp`` travel as FIELDS rather than being flattened into
    the content, so the turn keeps the speaker its first-person clauses anchor on
    and the time its relative dates resolve against. *timestamp* is the session
    anchor, used for messages that carry no time of their own.
    """
    turns: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        turns.append(
            {
                "role": str(msg.get("role") or "user"),
                "content": content,
                "timestamp": str(msg.get("time_anchor") or msg.get("timestamp") or timestamp),
            }
        )
    return turns
