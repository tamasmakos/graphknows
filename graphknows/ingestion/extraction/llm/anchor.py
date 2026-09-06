"""The response gates: anchor, vocabulary, confidence, then the local cleaning.

Four gates run in the order contracts/decoder.md §3 fixes, because the order is
what decides which counter a doubly-bad item lands in:

1. **anchor** — every entity surface and role filler is located by *exact*
   substring against the chunk, first unclaimed occurrence winning, so the same
   response always yields the same offsets (research.md R9). Relation endpoints
   are matched downstream by name, not by offset, so they are not anchored here;
   a relation's quoted ``evidence`` is checked for containment instead, and nulled
   when it is absent, because that string is the one the answer generator's fact
   sheet turns back into prompt text (FR-011).
2. **vocabulary** — labels, predicates, frames and role names are matched
   casefolded-exact against the request's closed sets. A near miss is a drop,
   never a mapping: an approximate match would manufacture the grounding the
   anchor law forbids (FR-013).
3. **confidence** — relations below ``decode_confidence_min`` are dropped. A
   threshold of ``0.0`` disables the gate, which is how FR-031's fourth ablation
   run is performed; no separate switch exists.
4. **map** — through the same ``_clean_surface`` / ``_reduce_speaker_span`` /
   ``_strip_determiner`` / ``_relation_reject`` chain the local decoder uses, so
   the graph's ENTITY merge keys are produced identically in both modes.

Every drop increments a named gate that reaches ``IngestResult.abstentions``
(FR-017); the private imports below are the point, not an accident.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from graphknows.ingestion.extraction.entities.extractor import (
    ExtractionResult,
    _build_relation,
    _clean_surface,
    _dedupe_entities,
    _reduce_speaker_span,
    _strip_determiner,
)
from graphknows.ingestion.extraction.llm.schema import (
    DecodedEntity,
    DecodedFrameInstance,
    DecodedItems,
    DecodedRelation,
    DecodeRequest,
    FrameCandidate,
)

logger = logging.getLogger(__name__)

UNANCHORABLE_SURFACE = "unanchorable_surface"
OUT_OF_VOCABULARY = "out_of_vocabulary"
LOW_CONFIDENCE_RELATION = "low_confidence_relation"
UNANCHORABLE_EVIDENCE = "unanchorable_evidence"
MALFORMED_ITEM = "malformed_item"

# The LLM returns no per-entity confidence, and ENTITY.score is a real column the
# overlap and merge-key passes rank on. A decoder that asserted the span at all
# asserted it fully; a made-up spread would be worse than a constant.
_ASSERTED = 1.0


class _Anchor:
    """Exact-substring offsets within one chunk, each occurrence claimed once.

    Claiming is what makes a repeated surface deterministic *and* stops two role
    fillers of one frame collapsing onto the same span (research.md R9). Finding
    and claiming are separate steps so an item the vocabulary gate goes on to
    reject does not hold a span the next item legitimately needs.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._claimed: list[tuple[int, int]] = []

    def find(self, surface: str) -> tuple[int, int]:
        """``(start, end)`` of the first unclaimed occurrence, or ``(-1, -1)``."""
        if not surface:
            return -1, -1
        start = self._text.find(surface)
        while start >= 0:
            end = start + len(surface)
            if not any(start < taken_end and taken < end for taken, taken_end in self._claimed):
                return start, end
            start = self._text.find(surface, start + 1)
        return -1, -1

    def claim(self, span: tuple[int, int]) -> None:
        """Take the span out of circulation for every later surface."""
        self._claimed.append(span)


def _normalized(value: str) -> str:
    """Whitespace-collapsed and casefolded: the module's one comparison form."""
    return " ".join(value.split()).casefold()


def _offered(value: str, vocabulary: Iterable[str]) -> str:
    """The offered term this value names casefolded-exactly, or ``""``.

    Returns the *offered* spelling rather than the returned one, so the graph
    stores the closed vocabulary's own label whatever case the model replied in.
    """
    folded = _normalized(value)
    return next(
        (term for term in vocabulary if _normalized(term) == folded),
        "",
    )


@dataclass(frozen=True)
class GatedDecode:
    """What survived one chunk's gates, and what each gate ate."""

    result: ExtractionResult
    abstentions: Counter[str]


class ResponseGates:
    """One chunk's request, held against the response it came back with.

    Bound to the request because every gate reads it: the closed vocabularies,
    the speaker the cleaning folds spans onto, and the text the anchor searches.
    """

    def __init__(self, request: DecodeRequest, confidence_min: float = 0.0) -> None:
        self._request = request
        self._confidence_min = confidence_min
        # _reduce_speaker_span keys on casefolded names, and an unresolved speaker
        # ("") must make it the identity function rather than fold on the empty string.
        self._speakers = (
            frozenset({request.speaker.casefold()}) if request.speaker.strip() else frozenset()
        )
        self._abstentions: Counter[str] = Counter()
        # Entities and role fillers anchor in separate scopes: one entity naming a
        # surface that occurs once must not starve the role filler that names it too.
        self._entity_anchor = _Anchor(request.text)
        self._filler_anchor = _Anchor(request.text)

    def apply(self, items: DecodedItems) -> GatedDecode:
        """Gate one decoded response into an ``ExtractionResult`` plus its drops."""
        self._abstentions = Counter()
        self._entity_anchor = _Anchor(self._request.text)
        self._filler_anchor = _Anchor(self._request.text)
        if items.malformed_item:
            self._abstentions[MALFORMED_ITEM] = items.malformed_item

        entities = self._entities(items.entities)
        return GatedDecode(
            result=ExtractionResult(
                entities=entities,
                relations=self._relations(items.relations, entities),
                frames=self._frames(items.frames),
            ),
            abstentions=self._abstentions,
        )

    def _entities(self, decoded: Iterable[DecodedEntity]) -> list[dict[str, Any]]:
        """Anchored, in-vocabulary spans, folded by the local decoder's own pass."""
        spans: list[dict[str, Any]] = []
        for entity in decoded:
            span = self._entity_anchor.find(entity.surface)
            if span[0] < 0:
                self._abstentions[UNANCHORABLE_SURFACE] += 1
                continue
            label = _offered(entity.label, self._request.entity_labels)
            if not label:
                self._abstentions[OUT_OF_VOCABULARY] += 1
                continue
            self._entity_anchor.claim(span)
            spans.append(
                {
                    "text": entity.surface,
                    "label": label,
                    "score": _ASSERTED,
                    "start": span[0],
                    "end": span[1],
                }
            )
        # The same pass the local decoder runs: cleaning, overlap resolution and
        # — load-bearing — dedupe on the graph's UNIQUE merge key, without which
        # one surface returned twice kills the whole document's bulk write.
        # No label_to_type map survives the request, so a span is typed by the
        # taxonomy alone; ENTITY.type is coalesced downstream either way.
        return _dedupe_entities(spans, {}, self._speakers)

    def _relations(
        self, decoded: Iterable[DecodedRelation], entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """In-vocabulary, confident triples built exactly as the local path builds them."""
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in decoded:
            predicate = _offered(relation.predicate, self._request.relation_spec)
            if not predicate:
                self._abstentions[OUT_OF_VOCABULARY] += 1
                continue
            if relation.confidence < self._confidence_min:
                self._abstentions[LOW_CONFIDENCE_RELATION] += 1
                continue
            built, reason = _build_relation(
                relation.head,
                predicate,
                relation.tail,
                relation.confidence,
                "llm",
                entities,
                speakers=self._speakers,
            )
            if built is None:
                # Structural, not a response gate: the local path logs these too.
                logger.debug("relation guard dropped %r: %s", relation.predicate, reason)
                continue
            key = (built["head"].casefold(), built["relation"], built["tail"].casefold())
            if key in seen:
                continue
            seen.add(key)
            # The confidence occupies the place the DeBERTa verifier's score
            # occupies today in the write path (FR-014, research.md R11).
            built["evidence"] = self._evidence(relation.evidence)
            built["verifier_score"] = relation.confidence
            out.append(built)
        return out

    def _evidence(self, evidence: str) -> str:
        """The quoted span when it occurs in the chunk, else ``""`` and a counted drop.

        The relation itself survives: its head, tail and predicate each passed
        their own gate, so dropping the triple would discard three grounded facts
        to punish one ungrounded decoration (research.md R13).
        """
        if not evidence.strip():
            return ""
        if _normalized(evidence) in _normalized(self._request.text):
            return evidence
        self._abstentions[UNANCHORABLE_EVIDENCE] += 1
        return ""

    def _frames(self, decoded: Iterable[DecodedFrameInstance]) -> list[dict[str, Any]]:
        """Role fillers for the frames the LOCAL pass offered — never a new frame."""
        candidates = {
            (candidate.frame.casefold(), candidate.trigger.casefold()): candidate
            for candidate in self._request.frame_candidates
        }
        out: list[dict[str, Any]] = []
        for instance in decoded:
            key = (instance.frame.strip().casefold(), instance.trigger.strip().casefold())
            candidate = candidates.get(key)
            if candidate is None:
                self._abstentions[OUT_OF_VOCABULARY] += 1
                continue
            roles = self._roles(instance, candidate)
            if not roles:
                continue
            out.append(
                {
                    "frame": candidate.frame,
                    "trigger": candidate.trigger,
                    "trigger_offset": candidate.trigger_offset,
                    "roles": roles,
                }
            )
        return out

    def _roles(
        self, instance: DecodedFrameInstance, candidate: FrameCandidate
    ) -> dict[str, list[str]]:
        """Core elements of this candidate only, filled with anchored surfaces."""
        core = [name for name, _ in candidate.core_elements]
        roles: dict[str, list[str]] = {}
        for role, fillers in instance.roles.items():
            name = _offered(role, core)
            if not name:
                self._abstentions[OUT_OF_VOCABULARY] += 1
                continue
            if kept := [f for filler in fillers if (f := self._filler(filler))]:
                roles.setdefault(name, []).extend(kept)
        return roles

    def _filler(self, surface: str) -> str:
        """One anchored role filler, cleaned into the graph's merge key, or ``""``."""
        span = self._filler_anchor.find(surface)
        if span[0] < 0:
            self._abstentions[UNANCHORABLE_SURFACE] += 1
            return ""
        self._filler_anchor.claim(span)
        # Role fillers become ENTITY nodes, so they are folded by the same chain
        # the entity and relation-endpoint paths use; the junk gate downstream in
        # _frame_instances is unchanged and still applies.
        return _strip_determiner(_reduce_speaker_span(_clean_surface(surface), self._speakers))
