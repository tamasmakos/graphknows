"""Is a turn worth extracting from? An opt-in gate for ``infer``.

Both primary entry points — ``server/mcp/tools/stm.py`` and
``Memory.ingest_memory`` — hardcode ``infer=True``; only the LangGraph adapter
opts into this gate, via ``infer=worth_extracting``. Buffering every turn with
``infer=True`` mints entities from greetings: measured
on LoCoMo conv-30, "Jon" never existed as a node while "Hey Jon", "Thanks, Jon!"
and "Oh no, Jon!" did, across 15 fragments. The decision is per-turn and purely
linguistic — nothing about it is dataset-specific — so it belongs here rather
than re-written in every integration.

A skipped turn is still stored, embedded and retrievable as text; it just mints
no entities. A wrong skip therefore costs graph coverage, never the evidence.
"""

from __future__ import annotations

from typing import Any


def _nlp() -> Any:
    """The shared spaCy pipeline, memoized by :func:`graphknows.nlp.load_spacy_model`."""
    from graphknows.nlp import load_spacy_model
    from graphknows.settings import get_settings

    return load_spacy_model(get_settings().spacy_model)


shared_nlp = _nlp


def worth_extracting(text: str) -> bool:
    """True when *text* predicates something about something.

    Naming a person does not count — "Hey Jon!" and "Mochi's vet is Dr. Nagy"
    BOTH contain a PERSON entity, and only the second asserts anything; the
    vocative has no clause at all.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    doc = _nlp()(stripped)

    subject = next((t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), None)
    if subject is None:
        # Speech drops subjects constantly ("Still working on opening a dance
        # studio"). An asserting verb plus a content noun is an elliptical clause.
        verb = next((t for t in doc if t.pos_ == "VERB"), None)
        noun = next((t for t in doc if t.pos_ == "NOUN"), None)
        return verb is not None and (noun is not None or bool(doc.ents))

    if subject.pos_ in {"NOUN", "PROPN"}:
        return True
    if any(t.pos_ == "NOUN" for t in doc):
        return True
    # A first-person subject IS nameable when the speaker is carried as metadata
    # — "I moved to Budapest" is a fact about whoever said it. Rejecting it as
    # "a bare pronoun" is why a speaker-prefixed arm extracted 98% of turns to a
    # clean arm's 91%: the prefix handed every turn a PROPN subject.
    return subject.text.lower() in {"i", "we", "you", "my", "our"}


__all__ = ["worth_extracting"]
