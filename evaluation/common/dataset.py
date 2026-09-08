"""Text/date helpers shared by the benchmark dataset modules.

Used by ``evaluation/{beam,locomo,longmem}/dataset.py`` when normalizing raw
benchmark records into :class:`~evaluation.common.datamodels.EvalDocument`.
"""

from __future__ import annotations

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
