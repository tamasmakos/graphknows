"""Domain-level normalization helpers for labels and entity names."""

from __future__ import annotations

import re


def standardize_label(text: str) -> str:
    """Standardize label/entity names to UPPER_CASE_WITH_UNDERSCORES."""
    if not text:
        return text
    text = re.sub(r"[^\w]+", "_", text)
    text = text.strip("_")
    return text.upper()
