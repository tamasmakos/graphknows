"""Lazy spaCy sentence segmentation for chunking.

Uses a blank English pipeline with only the rule-based ``sentencizer`` — no
statistical model download, fast, and dependency-light. Splitting on sentence
boundaries keeps words and sentences intact when chunking long text.
"""

from __future__ import annotations

import re
from typing import Any

_nlp = None


def _get_sentencizer() -> Any:
    """Return a cached blank spaCy pipeline with a sentencizer component."""
    global _nlp
    if _nlp is None:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        # Dialogue lines can be long; lift the default 1M-char cap defensively.
        nlp.max_length = 5_000_000
        _nlp = nlp
    return _nlp


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, preserving newline-delimited lines as boundaries.

    Each source line (e.g. one chat message) is segmented independently so a
    sentence never spans two messages, then sentences are flattened in order.
    Empty fragments are dropped.
    """
    text = text.strip()
    if not text:
        return []
    nlp = _get_sentencizer()
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        doc = nlp(line)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                sentences.append(s)
    # Fallback: if the sentencizer produced nothing usable, return the whole text.
    return sentences or [re.sub(r"\s+", " ", text).strip()]
