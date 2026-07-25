"""Relation verifier gate — semantic curation before the graph merge.

The relex extractor pairs co-occurring spans confidently even when the text
expresses no relation between them ("Dancing --DO_WITH--> dance", "We all
--NEED--> push"); relex's own score can't catch these because it *is* confident.
This gate scores each (head, relation, tail) with the
``oneryalcin/gliner2-relation-verifier`` DeBERTa-v3 classifier and drops the
ones below ``threshold`` before they reach the graph.

Forked behind a factory like the LLM relation extractor: enabled builds
:class:`RelationVerifierGate`, disabled returns :class:`NullRelationVerifier`,
so the extractor calls ``filter`` unconditionally. Load failure (offline, no
HF access) falls back to the null object — a missing model never kills ingest.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings

log = logging.getLogger(__name__)


class NullRelationVerifier:
    """No-op gate — passes every relation through unchanged."""

    def filter(self, text: str, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *relations* untouched."""
        return relations


class RelationVerifierGate:
    """Drop relex relations whose head/tail pair fails semantic verification.

    The DeBERTa encoder + MLP load lazily on first ``filter`` call (~1 forward
    pass per chunk plus a cheap MLP per relation). The underlying model is not
    reentrant, so inference is serialised — this gate is shared across
    concurrently-ingesting conversations exactly like the relex model.
    """

    def __init__(self, model_id: str, threshold: float) -> None:
        self._model_id = model_id
        self._threshold = threshold
        self._verifier: Any | None = None
        self._lock = threading.Lock()
        self._disabled = False

    def _ensure_loaded(self) -> bool:
        """Load the verifier once; disable the gate on failure. Returns readiness."""
        if self._verifier is not None:
            return True
        if self._disabled:
            return False
        with self._lock:
            if self._verifier is not None:
                return True
            if self._disabled:
                return False
            try:
                from graphknows.ingestion.extraction.relations._gliner2_verifier import (
                    RelationVerifier,
                )

                log.info("loading relation verifier %s (first call)...", self._model_id)
                self._verifier = RelationVerifier.from_pretrained(
                    self._model_id, threshold=self._threshold
                )
            except Exception as exc:
                log.warning(
                    "relation verifier unavailable (%s); passing relations through: %s",
                    self._model_id,
                    exc,
                )
                self._disabled = True
                return False
        return True

    def filter(self, text: str, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only relations that verify at or above ``threshold``.

        Head/tail char offsets (which the extractor strips during cleaning) are
        recovered by locating each surface in *text*. A relation whose surface
        cannot be located is kept — it is unverifiable, not disproven, so we do
        not silently delete a fact we merely failed to offset.
        """
        if not relations or not self._ensure_loaded():
            return relations
        assert self._verifier is not None

        grouped: dict[str, list[dict[str, Any]]] = {}
        passthrough: list[dict[str, Any]] = []
        for rel in relations:
            head, tail = rel["head"], rel["tail"]
            hs, ts = text.find(head), text.find(tail)
            if hs < 0 or ts < 0:
                passthrough.append(rel)
                continue
            grouped.setdefault(rel["relation"], []).append(
                {
                    "head": {"text": head, "start": hs, "end": hs + len(head)},
                    "tail": {"text": tail, "start": ts, "end": ts + len(tail)},
                    "_ref": rel,
                }
            )
        if not grouped:
            return relations

        try:
            verified = self._verifier.verify(text, grouped)
        except Exception as exc:
            log.warning("relation verifier errored on chunk; passing through: %s", exc)
            return relations

        kept = list(passthrough)
        for instances in verified.values():
            for inst in instances:
                rel = inst["_ref"]
                rel["verifier_score"] = round(float(inst.get("verifier_score", 0.0)), 4)
                kept.append(rel)
        return kept


def build_relation_verifier(
    settings: GraphKnowsSettings,
) -> RelationVerifierGate | NullRelationVerifier:
    """Return the verifier gate when enabled, else the null object."""
    if not settings.relation_verifier:
        return NullRelationVerifier()
    return RelationVerifierGate(
        settings.relation_verifier_model, settings.relation_verifier_threshold
    )


__all__ = [
    "NullRelationVerifier",
    "RelationVerifierGate",
    "build_relation_verifier",
]
