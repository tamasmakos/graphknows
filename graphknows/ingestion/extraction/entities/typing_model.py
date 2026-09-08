"""Definition-described entity typing: an ontology class definition IS the label hint.

The relex model the main extractor wraps takes a flat list of label STRINGS, so
every ontology definition the pipeline carefully assembles is dropped at the call
boundary — the model only ever sees ``"Act"``, never what CCO means by it. That
costs exactly the discriminating asset: measured on six LoCoMo windows, typing
against definitions moved ``research`` from OCCUPATION to **Act** and ``painting``
from WORK_OF_ART to **Artifact**, and lifted ontology-typed coverage from 14/24 to
22/26 surfaces.

GLiNER2's own schema API does take ``{label: description}``, so this module runs a
SECOND, smaller pass whose whole purpose is that description channel. It is
deliberately not folded into the main extractor: relex owns relations and spans
and answers to a different API, and giving one wrapper two model backends to keep
straight is how both end up wrong.

Per LINE, not per chunk — the line keeps its ``Speaker:`` prefix, which is what
resolves "I". Handing the model a five-turn window attributed one speaker's
sentences to another.

DEFAULT OFF (``GRAPHKNOWS_DEFINITION_TYPING``): it is a second ~0.5B model
resident per process and a forward pass per line.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from graphknows.settings import get_settings

log = logging.getLogger(__name__)

_model: Any = None

# The whole schema is ONE encoder input and the labels compete for its 512-token
# window, so a definition is truncated rather than allowed to crowd its
# neighbours out. Measured band, not a guess: the prototype ran this at 140.
MAX_DEFINITION: int = 140

# Span floor. Lower than the relex default on purpose — a definition-described
# label is a harder target than a bare type name, so scores run lower for the
# same span.
DEFAULT_THRESHOLD: float = 0.4


def typing_enabled() -> bool:
    """Whether the definition-described typing pass runs (default off).

    Read at call time so an eval A/B can flip it per subprocess, matching
    ``GRAPHKNOWS_FRAME_SRL`` and ``GRAPHKNOWS_RELATION_GROUP_RANKING``.
    """
    return get_settings().definition_typing


def _load() -> Any:
    """The typing model, loaded once per process."""
    global _model
    if _model is None:
        from gliner2 import GLiNER2

        settings = get_settings()
        t0 = __import__("time").monotonic()
        _model = GLiNER2.from_pretrained(settings.typing_model)
        if device := settings.typing_device.strip():
            _model = _model.to(device)
        log.info(
            "typing model %s loaded in %.1fs",
            settings.typing_model,
            __import__("time").monotonic() - t0,
        )
    return _model


def build_schema(terms: Any, labels: Sequence[str]) -> dict[str, str]:
    """``{class label: definition}`` for *labels*, in the order given.

    The definition is the point; a class whose definition is empty falls back to
    its own label so the slot still exists rather than silently disappearing.
    """
    by_label = {t.label.casefold(): t for t in getattr(terms, "classes", [])}
    schema: dict[str, str] = {}
    for label in labels:
        term = by_label.get(label.casefold())
        if term is None:
            continue
        schema[term.label] = (term.definition or term.label)[:MAX_DEFINITION]
    return schema


def type_lines(
    lines: list[str], schema: dict[str, str], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, tuple[str, float]]:
    """Type each line against *schema*: ``{surface: (class label, confidence)}``.

    Keyed by the surface AS WRITTEN, deduplicated case-insensitively. The
    distinction matters downstream: an ENTITY merges on a casefolded key but
    DISPLAYS its ``name``, so keying the result casefolded would mint "melanie"
    for any surface relex did not already have.

    Blocking — the caller runs it off the event loop. The first class to claim a
    surface keeps it: re-typing the same string differently later in a chunk is
    the model disagreeing with itself, and the earlier line is the one whose
    speaker context is closest to the mention.

    Confidences are the model's own. The prototype threw them away and wrote a
    flat 1.0, which makes a hedged guess indistinguishable from a certain one
    downstream where ``INSTANCE_OF.score`` is all a reader has.
    """
    if not lines or not schema:
        return {}
    model = _load()
    out: dict[str, tuple[str, float]] = {}
    seen: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            found = model.extract_entities(
                line, dict(schema), threshold=threshold, include_confidence=True
            )
        except Exception as exc:  # pragma: no cover - a bad line must not kill the chunk
            log.warning("definition typing failed on a line (%s)", exc)
            continue
        for label, spans in (found.get("entities") or {}).items():
            for span in spans or []:
                surface, score = _span(span)
                if surface and surface.casefold() not in seen:
                    seen.add(surface.casefold())
                    out[surface] = (str(label), score)
    return out


def _span(span: Any) -> tuple[str, float]:
    """``(surface, confidence)`` from a span, whatever shape the model returned.

    ``include_confidence`` changes the element from a bare string to a mapping
    (or a pair), and the shape has moved between gliner2 releases — so read it
    defensively and fall back to the threshold-implied floor rather than
    inventing a 1.0.
    """
    if isinstance(span, str):
        return span.strip(), DEFAULT_THRESHOLD
    if isinstance(span, dict):
        text = str(span.get("text") or span.get("span") or span.get("value") or "").strip()
        raw = span.get("confidence", span.get("score"))
        return text, float(raw) if isinstance(raw, int | float) else DEFAULT_THRESHOLD
    if isinstance(span, list | tuple) and span:
        text = str(span[0]).strip()
        raw = span[1] if len(span) > 1 else None
        return text, float(raw) if isinstance(raw, int | float) else DEFAULT_THRESHOLD
    return "", 0.0
