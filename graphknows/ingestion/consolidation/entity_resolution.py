"""Identity: which entities are one, on what evidence, and reversibly.

Three things keep this honest. The **mention layer** is the input — an entity is
what its mentions say it is, so its surfaces stay attached to their segments and
its type is a histogram of every label it was read under, never the first one
seen (FR-016, FR-021). The **ladder** is the decision — veto, then strong, then
weak — and it stops at the first rung that answers: a pack veto or disjoint
sibling concepts forbid both merge and candidate (FR-019), an identical
normalised key within a block with agreeing concepts merges (FR-017), and
anything weaker is a scored candidate the recall path commits (FR-018). The
**plan** is the output: every :class:`Link` of a batch is decided before a
single one is applied, so the ``MERGED_INTO`` log exists before the structural
change it records and a merge is replayable — and undoable — from the log alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from graphknows.ingestion.consolidation.blocking import block_key
from graphknows.ingestion.consolidation.candidates import bounded_candidates
from graphknows.models.fact import Mention
from graphknows.models.report import Counters


class Layer(StrEnum):
    """The rung of the ladder that decided a pair — recorded on every link."""

    VETO = "veto"
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


@dataclass(frozen=True)
class EntityView:
    """One entity as the mention layer sees it: its surfaces and its tallies.

    Attributes:
        id: The entity's normalised-key id.
        name_norm: The resolver's normalised name.
        surfaces: Every surface form observed, in the order first read. Kept as
            evidence-bearing surfaces, never flattened into an alias string.
        type_histogram: Observations per pack label (FR-021).
        concepts: Concept uris the entity's labels resolved to.
        embedding: The entity's definition vector, the weak rung's only signal.
    """

    id: str
    name_norm: str
    surfaces: tuple[str, ...] = ()
    type_histogram: Mapping[str, int] = field(default_factory=dict)
    concepts: frozenset[str] = frozenset()
    embedding: tuple[float, ...] = ()

    @property
    def block_key(self) -> str:
        """The block this entity is compared within (FR-020)."""
        return block_key(self.name_norm, self.type_histogram)

    @property
    def observations(self) -> int:
        """How often the entity was mentioned — the merge direction's tiebreak."""
        return sum(self.type_histogram.values())


def mention_layer(mentions: Iterable[Mention]) -> dict[str, EntityView]:
    """The entities *mentions* refer to, keyed by id.

    Nothing is resolved here: this is the bottom-up half, folding surfaces and
    labels into one view per entity so the ladder reads observations rather
    than a single extractor's first guess.
    """
    surfaces: dict[str, list[str]] = {}
    histograms: dict[str, dict[str, int]] = {}
    for mention in mentions:
        seen = surfaces.setdefault(mention.entity_id, [])
        if mention.surface not in seen:
            seen.append(mention.surface)
        if mention.label:
            histogram = histograms.setdefault(mention.entity_id, {})
            histogram[mention.label] = histogram.get(mention.label, 0) + 1
    return {
        entity_id: EntityView(
            id=entity_id,
            name_norm=entity_id,
            surfaces=tuple(forms),
            type_histogram=histograms.get(entity_id, {}),
        )
        for entity_id, forms in surfaces.items()
    }


@dataclass(frozen=True)
class Verdict:
    """What the ladder concluded about a pair, before it is given a direction."""

    layer: Layer
    evidence: str
    score: float


@dataclass(frozen=True)
class Link:
    """One decided pair, as both merge log and candidate plane record it.

    A ``STRONG`` link is a ``MERGED_INTO`` log entry, a ``WEAK`` one a
    ``SAME_AS_CANDIDATE`` edge; both carry the layer, the evidence and the time,
    which is everything a replay needs to reconstruct — or reverse — the merge.
    """

    source_id: str
    target_id: str
    layer: Layer
    evidence: str
    score: float
    at: datetime


@dataclass(frozen=True)
class ResolutionPlan:
    """Every decision of one batch, complete before any of it is applied.

    Attributes:
        merges: The ``MERGED_INTO`` log, oldest decision first.
        candidates: Scored same-as candidates, left for recall to commit.
        vetoed: Pairs the veto rung stopped.
        truncated: Candidates the comparison cap dropped.
    """

    merges: tuple[Link, ...] = ()
    candidates: tuple[Link, ...] = ()
    vetoed: int = 0
    truncated: int = 0

    @property
    def counters(self) -> Counters:
        """This plan as the R9 counter set ingest reports."""
        return Counters(
            merges_committed=len(self.merges),
            candidate_edges_written=len(self.candidates),
            merges_vetoed=self.vetoed,
            candidates_truncated=self.truncated,
        )


def _no_veto(a: str, b: str) -> bool:
    """The identity veto of a namespace with no pack rule: it forbids nothing."""
    return False


@dataclass(frozen=True)
class Resolver:
    """The ladder, holding the rules it climbs with.

    Attributes:
        veto: The pack's identity veto: True when these two ids never merge.
        disjoint: Concept pairs that cannot describe one thing.
        floor: Cosine below which the embedding says nothing.
        margin: How far past the floor a cosine must reach to be evidence.
    """

    veto: Callable[[str, str], bool] = _no_veto
    disjoint: frozenset[frozenset[str]] = frozenset()
    floor: float = 0.80
    margin: float = 0.05

    def resolve(self, left: EntityView, right: EntityView) -> Link:
        """The first rung that answers for this pair, as a directed link."""
        return _link((left, right), self.verdict(left, right))

    def verdict(self, left: EntityView, right: EntityView) -> Verdict:
        """The first rung of the ladder that answers for this pair.

        The ladder is ordered by what it costs to be wrong: a veto is cheap and
        final, a merge is the expensive commitment, a candidate defers.
        """
        if self.veto(left.id, right.id) or self._disjoint(left, right):
            return Verdict(Layer.VETO, "veto", 0.0)
        if self._strong(left, right):
            return Verdict(Layer.STRONG, f"key={left.name_norm}", 1.0)
        similarity = _cosine(left.embedding, right.embedding)
        if shared := left.concepts & right.concepts:
            evidence = f"concept={','.join(sorted(shared))}"
            return Verdict(Layer.WEAK, evidence, max(similarity, self.floor))
        if similarity >= self.floor + self.margin:
            return Verdict(Layer.WEAK, f"cosine={similarity:.3f}", similarity)
        return Verdict(Layer.NONE, "", similarity)

    def plan(
        self,
        views: Mapping[str, EntityView],
        neighbours: Mapping[str, Sequence[str]],
    ) -> ResolutionPlan:
        """Resolve every entity in *views* against the neighbours proposed for it.

        *neighbours* is the proposal stream of blocking and the vector index, in
        priority order; the cap in :mod:`.candidates` is what keeps this off a
        full-table scan (FR-020). Nothing is written here — the plan is the
        whole batch's decisions, so the log precedes every structural change.
        """
        merges: list[Link] = []
        candidates: list[Link] = []
        decided: set[tuple[str, str]] = set()
        vetoed = truncated = 0
        for entity_id, view in views.items():
            pool = bounded_candidates(entity_id, neighbours.get(entity_id, ()))
            truncated += pool.truncated
            for other_id in pool.ids:
                other = views.get(other_id)
                # A pair is decided once: the reverse proposal is the same pair.
                pair = (min(entity_id, other_id), max(entity_id, other_id))
                if other is None or pair in decided:
                    continue
                decided.add(pair)
                link = self.resolve(view, other)
                if link.layer is Layer.STRONG:
                    merges.append(link)
                elif link.layer is Layer.WEAK:
                    candidates.append(link)
                elif link.layer is Layer.VETO:
                    vetoed += 1
        return ResolutionPlan(tuple(merges), tuple(candidates), vetoed, truncated)

    def _disjoint(self, left: EntityView, right: EntityView) -> bool:
        """True when the pair's concepts include a disjoint sibling pair."""
        return any(
            frozenset((one, other)) in self.disjoint
            for one in left.concepts
            for other in right.concepts
            if one != other
        )

    def _strong(self, left: EntityView, right: EntityView) -> bool:
        """Identical key within one block, under concepts that do not disagree."""
        if left.name_norm != right.name_norm or left.block_key != right.block_key:
            return False
        return not (left.concepts and right.concepts) or bool(left.concepts & right.concepts)


def replay(log: Iterable[Link]) -> dict[str, str]:
    """Where each merged id ends up, folded out of the ``MERGED_INTO`` log alone.

    The log is replayed oldest decision first, then every chain is followed to
    its end, so the result is the surviving id for each id the batch merged
    away — the merge reconstructed without the entities it was decided from
    (SC-007).
    """
    merged = {link.source_id: link.target_id for link in sorted(log, key=lambda e: e.at)}
    return {source: _survivor(merged, source) for source in merged}


def _survivor(merged: Mapping[str, str], source: str) -> str:
    """The end of *source*'s merge chain: the id no entry of the log merged away."""
    seen = source
    for _ in range(len(merged)):
        if (further := merged.get(seen)) is None:
            break
        seen = further
    return seen


def undo(log: Sequence[Link], entry: Link) -> dict[str, str]:
    """The same replay with *entry* dropped: one merge reversed from the log alone."""
    return replay(link for link in log if link != entry)


def _link(pair: tuple[EntityView, EntityView], verdict: Verdict) -> Link:
    """*verdict* given a direction: the better-observed side survives the merge.

    Direction is decided by observations then id, never by which view arrived
    first — a batch reordered by the store must plan the same merge.
    """
    source, target = sorted(pair, key=lambda view: (view.observations, view.id))
    return Link(
        source_id=source.id,
        target_id=target.id,
        layer=verdict.layer,
        evidence=verdict.evidence,
        score=verdict.score,
        at=datetime.now(UTC),
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of two vectors, or ``0.0`` when either says nothing."""
    if len(left) != len(right) or not left:
        return 0.0
    norms = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right, strict=True)) / norms if norms else 0.0
