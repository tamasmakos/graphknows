"""What ingest and recall hand back — with the counters that make them checkable.

Both results carry the same counter set (research.md R9): a channel that skips,
truncates or vetoes says so in a named field rather than logging it and
returning a bare collection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from graphknows.models.fact import Fact


class Counters(BaseModel, frozen=True):
    """The named counters of research.md R9 — one set, ingest and recall.

    Every counter defaults to zero, so a caller reads "nothing was skipped"
    rather than "nothing was reported".
    """

    # Ingest.
    sources_deduplicated: int = 0
    files_zero_segments: int = 0
    records_skipped_unknown_type: int = 0
    blocks_skipped_unknown_type: int = 0
    records_skipped_malformed: int = 0
    records_skipped_duplicate: int = 0
    time_anchor_inferred: int = 0
    candidates_truncated: int = 0
    merges_committed: int = 0
    candidate_edges_written: int = 0
    merges_vetoed: int = 0
    ingest_queue_wait_ms: int = 0

    # Recall.
    symbols_resolved: int = 0
    symbols_unresolved: int = 0
    facts_returned: int = 0
    facts_truncated_by_budget: int = 0
    facts_excluded_tombstoned: int = 0
    pool_below_floor: int = 0


class IngestReport(BaseModel, frozen=True):
    """What one ingest wrote, and everything it declined to write.

    Attributes:
        source_id: Identity of the ingested source.
        segments_written: Evidence segments committed.
        facts_written: Facts committed.
        entities_touched: Entities created or merged into.
        counters: The R9 counter set for this run.
    """

    source_id: str
    segments_written: int = 0
    facts_written: int = 0
    entities_touched: int = 0
    counters: Counters = Field(default_factory=Counters)


class RecallBudget(BaseModel, frozen=True):
    """The explicit bound on a recalled slice (FR-010).

    Attributes:
        max_facts: Ceiling on the facts returned.
        max_evidence_per_fact: Ceiling on the evidence segments per fact; at
            least one, since a fact without evidence is never returned.
    """

    max_facts: int = Field(default=20, gt=0)
    max_evidence_per_fact: int = Field(default=1, gt=0)


class Evidence(BaseModel, frozen=True):
    """The segment a fact was read from, quoted where it was read.

    Attributes:
        source_uri: The source the segment belongs to.
        byte_range: ``(start, end)`` byte offsets within that source.
        text: The segment text as written.
    """

    source_uri: str
    byte_range: tuple[int, int]
    text: str


class FactWithEvidence(BaseModel, frozen=True):
    """A fact and the segments that assert it — never one without the other.

    A fact that cannot produce evidence is a write-time defect (SC-002), so an
    empty ``evidence`` list is rejected here rather than returned.
    """

    fact: Fact
    evidence: list[Evidence] = Field(min_length=1)


class RecallResult(BaseModel, frozen=True):
    """A budgeted subgraph slice, with what the budget and the gates removed.

    Attributes:
        facts: The slice, each fact with its evidence.
        no_evidence: "Nothing is known", as opposed to a lookup that failed
            (FR-015). Required whenever ``facts`` is empty.
        budget: The budget that was applied.
        truncated_by: The ordering that cut the slice, when it was cut.
        counters: The R9 counter set for this recall.
    """

    facts: list[FactWithEvidence] = Field(default_factory=list)
    no_evidence: bool = False
    budget: RecallBudget = Field(default_factory=RecallBudget)
    truncated_by: str | None = None
    counters: Counters = Field(default_factory=Counters)

    @model_validator(mode="after")
    def _emptiness_is_stated(self) -> RecallResult:
        if not self.facts and not self.no_evidence:
            raise ValueError("empty recall must set no_evidence")
        return self
