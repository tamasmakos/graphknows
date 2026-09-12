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

Coverage — which edges this gate is responsible for:

* Applied per REL tier at the ingest WRITE FUNNEL
  (``_STMIngestHandler._write_relations``), not at one producer: relex triples
  are scored inside the extractor where their text and offsets are freshest,
  DSPy triples are scored at the funnel with that same already-loaded gate. So
  every REL edge ingest writes has passed a gate.
* The frame layer is deliberately OUT of scope.
  ``GraphStore.write_frame_instances`` writes reified
  ``(FRAME_INSTANCE)-[:PLAYS_ROLE {role}]->(ENTITY)`` edges — a role filler is
  not a (head, predicate, tail) triple, so this classifier has nothing to
  score; those edges carry ``match_sim``/``role_confidence`` as their own
  quality signal. The flat ENTITY→ENTITY co-role projection that WOULD have
  been verifiable was removed (see ``write_frame_instances``' docstring), so
  this is a labelled gap, not an oversight.
* Every producer is gated at the funnel, so a future producer that skips the
  gate writes an unscored edge — there is no flush-time coverage sweep behind
  it to catch one.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings

log = logging.getLogger(__name__)


def _claim_span(text: str, surface: str, cursors: dict[str, int]) -> tuple[int, int] | None:
    """Locate *surface* in *text*, claiming successive occurrences per call.

    A repeated surface advances past the occurrence claimed for it last time,
    so two relations that mention the same text at two different points don't
    both anchor to the first one. Once every occurrence is claimed, later
    calls wrap back to the first — so two relations that share the text's
    only mention of *surface* (e.g. one's tail is the other's head) both
    resolve to it instead of one going unlocatable.
    """
    start = cursors.get(surface, 0)
    idx = text.find(surface, start)
    if idx < 0 and start:
        idx = text.find(surface)
    if idx < 0:
        return None
    cursors[surface] = idx + 1
    return idx, idx + len(surface)


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
        recovered by claiming each surface's occurrence in *text* (see
        :func:`_claim_span`) rather than always anchoring to its first match. A
        relation whose surface cannot be located at all is kept — it is
        unverifiable, not disproven, so we do not silently delete a fact we
        merely failed to offset.
        """
        if not relations or not self._ensure_loaded():
            return relations
        assert self._verifier is not None

        grouped: dict[str, list[dict[str, Any]]] = {}
        passthrough: list[dict[str, Any]] = []
        cursors: dict[str, int] = {}
        for rel in relations:
            head, tail = rel["head"], rel["tail"]
            head_span = _claim_span(text, head, cursors)
            tail_span = _claim_span(text, tail, cursors)
            if head_span is None or tail_span is None:
                passthrough.append(rel)
                continue
            hs, he = head_span
            ts, te = tail_span
            grouped.setdefault(rel["relation"], []).append(
                {
                    "head": {"text": head, "start": hs, "end": he},
                    "tail": {"text": tail, "start": ts, "end": te},
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
