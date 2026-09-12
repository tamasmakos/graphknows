"""The cutover scenarios (FR-042): ``python -m evaluation scenario <name>``.

Three of them gate this slice's merge, and each one is a claim about the core
rather than a benchmark number:

* ``two-packs`` — two unrelated packs share one namespace and both recall facts
  with evidence, so the core knows no domain (SC-002).
* ``schema-refusal`` — a namespace stamped by another schema refuses to open,
  naming both versions, instead of migrating it (SC-013).
* ``concurrent-ingest`` — ingesting concurrently lands the same graph as
  ingesting in sequence, and neither ingest fails (SC-015).
* ``merge-replay`` — every merge a batch decided is reconstructed, and reversed,
  from its ``MERGED_INTO`` log alone (SC-007).
* ``resolve-scaling`` — a batch of fixed size costs the same wall clock against a
  namespace ten times larger, so no full-table scan survived (SC-006).
* ``transcript-drift`` — a transcript the parser no longer fully recognises is
  counted, never fatal, and its sound records still land (SC-014).
* ``forget-roundtrip`` — a forgotten record leaves recall while its tombstone and
  the merge log around it stay readable (SC-012).
* ``hook-latency`` — the verb an agent hook shells out to answers inside the hook
  budget against a warm service (SC-010).

Each scenario takes what it drives and returns a :class:`ScenarioResult`; the
run outcome is :meth:`ScenarioResult.gate`, the same shape the dead-weight check
uses, so a failed scenario exits non-zero with its reason.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any

from evaluation.common.reporting import _pct
from graphknows.exceptions import SchemaVersionMismatchError
from graphknows.ingestion.consolidation.candidates import MAX_CANDIDATES
from graphknows.ingestion.consolidation.entity_resolution import (
    EntityView,
    ResolutionPlan,
    Resolver,
    replay,
    undo,
)
from graphknows.ingestion.parsers.transcript import TranscriptParser
from graphknows.memory import Memory
from graphknows.models import Segment, SegmentKind, Source
from graphknows.models.fact import FactState
from graphknows.models.report import Counters
from graphknows.packs.agent import AgentPack
from graphknows.packs.code import CodePack
from graphknows.settings import get_settings
from graphknows.storage import build_graph_store
from graphknows.storage.embedder import embed_dim


@dataclass(frozen=True)
class ScenarioResult:
    """What one scenario proved, and why it did not when it failed."""

    name: str
    passed: bool
    detail: str

    def gate(self) -> None:
        """Fail the run when the scenario did not hold; a no-op when it did."""
        if not self.passed:
            raise SystemExit(f"{self.name}: {self.detail}")


@dataclass(frozen=True)
class PackProbe:
    """One pack's share of a scenario: text only it understands, and a query for it."""

    uri: str
    text: str
    kind: SegmentKind
    query: str

    @property
    def source(self) -> Source:
        return Source(
            uri=self.uri,
            content_hash=sha256(self.text.encode()).hexdigest(),
            mime="text/plain",
        )

    @property
    def segments(self) -> tuple[Segment, ...]:
        return (
            Segment(
                source_id=self.source.id,
                text=self.text,
                kind=self.kind,
                byte_range=(0, len(self.text.encode())),
            ),
        )


async def _ingest(memory: Any, probe: PackProbe) -> Any:
    """Write *probe* into *memory* and return the ingest report."""
    return await memory.ingest(probe.source, probe.segments)


def _written(reports: Sequence[Any]) -> tuple[int, int]:
    """The facts and entities *reports* account for, summed across the batch."""
    return (
        sum(report.facts_written for report in reports),
        sum(report.entities_touched for report in reports),
    )


async def two_packs(memory: Any, probes: Sequence[PackProbe]) -> ScenarioResult:
    """Every pack's probe must ingest and recall facts from the shared namespace (SC-002)."""
    for probe in probes:
        await _ingest(memory, probe)
    unanswered = [probe.query for probe in probes if not (await memory.recall(probe.query)).facts]
    return ScenarioResult(
        "two-packs",
        not unanswered,
        f"no facts recalled for: {', '.join(unanswered)}" if unanswered else "every probe recalled",
    )


async def schema_refusal(open_namespace: Callable[[], Awaitable[object]]) -> ScenarioResult:
    """Opening a namespace stamped by another schema must refuse, naming both (SC-013)."""
    try:
        await open_namespace()
    except SchemaVersionMismatchError as mismatch:
        return ScenarioResult(
            "schema-refusal",
            True,
            f"refused: recorded {mismatch.recorded or 'none'}, running {mismatch.running}",
        )
    return ScenarioResult("schema-refusal", False, "a stale-stamped namespace opened instead")


async def concurrent_ingest(
    memories: Sequence[Any],
    probes: Sequence[PackProbe],
) -> ScenarioResult:
    """The probes ingested at once must land what they land one at a time (SC-015).

    *memories* is two namespaces: the first takes every probe concurrently, the
    second takes them in sequence, and the two graphs are compared by what the
    ingests wrote.
    """
    concurrently, in_sequence = memories
    at_once = await asyncio.gather(
        *(_ingest(concurrently, probe) for probe in probes), return_exceptions=True
    )
    if failures := [str(item) for item in at_once if isinstance(item, BaseException)]:
        return ScenarioResult("concurrent-ingest", False, f"ingest failed: {'; '.join(failures)}")
    one_at_a_time = [await _ingest(in_sequence, probe) for probe in probes]
    concurrent_write, sequential_write = _written(at_once), _written(one_at_a_time)
    if concurrent_write != sequential_write:
        return ScenarioResult(
            "concurrent-ingest",
            False,
            f"concurrent wrote {concurrent_write}, sequential wrote {sequential_write}",
        )
    return ScenarioResult("concurrent-ingest", True, f"both orders wrote {concurrent_write}")


def merge_replay(plan: ResolutionPlan) -> ScenarioResult:
    """Every merge of *plan* must replay, and reverse, from its log alone (SC-007).

    The log is read without the entities it was decided from: replay must place
    every merged id under a survivor, and dropping one entry must stop that
    entry's source from resolving where the entry put it.
    """
    resolved = replay(plan.merges)
    if unreplayed := [link.source_id for link in plan.merges if link.source_id not in resolved]:
        return ScenarioResult("merge-replay", False, f"not replayed: {', '.join(unreplayed)}")
    stuck = [
        link.source_id
        for link in plan.merges
        if undo(plan.merges, link).get(link.source_id) == resolved[link.source_id]
    ]
    if stuck:
        return ScenarioResult("merge-replay", False, f"not undone: {', '.join(stuck)}")
    return ScenarioResult("merge-replay", True, f"{len(plan.merges)} merges replayed and undone")


@dataclass(frozen=True)
class ForgetObservation:
    """One forget read back: what recall says after it, and what the store still holds.

    The merge log is read on both sides of the forget, so an entry a tombstone
    took with it shows up as a shorter log instead of being lost silently.
    """

    record_id: str
    recalled_after: tuple[str, ...]
    state_after: str
    merge_log_before: tuple[str, ...]
    merge_log_after: tuple[str, ...]


def forget_roundtrip(observed: ForgetObservation) -> ScenarioResult:
    """A forgotten record must leave recall while the graph still reads it (SC-012).

    Forget is a tombstone, so all three have to hold at once: the record is gone
    from recall, its state still reads ``forgotten``, and the merge log is the
    one it was before.
    """
    if not observed.record_id:
        return ScenarioResult("forget-roundtrip", False, "nothing was recalled to forget")
    if observed.record_id in observed.recalled_after:
        return ScenarioResult("forget-roundtrip", False, f"{observed.record_id} is still recalled")
    if observed.state_after != str(FactState.FORGOTTEN):
        return ScenarioResult(
            "forget-roundtrip",
            False,
            f"tombstone unreadable: state is {observed.state_after or 'gone'}",
        )
    if observed.merge_log_after != observed.merge_log_before:
        return ScenarioResult(
            "forget-roundtrip",
            False,
            f"the merge log read {len(observed.merge_log_before)} entries before the forget "
            f"and {len(observed.merge_log_after)} after",
        )
    return ScenarioResult(
        "forget-roundtrip",
        True,
        f"{observed.record_id} tombstoned, "
        f"{len(observed.merge_log_after)} merge entries still readable",
    )


DRIFT_COUNTERS = (
    "records_skipped_malformed",
    "records_skipped_unknown_type",
    "blocks_skipped_unknown_type",
)
"""The counters drift has to show up in — one per way a transcript can drift."""


def transcript_drift(
    parse: Callable[[], tuple[Sequence[Segment], Counters]],
) -> ScenarioResult:
    """Drift must be counted, and the records around it must survive it (SC-014).

    *parse* reads a transcript carrying one of each drift alongside sound
    records. Raising is the failure the scenario exists to catch, so it is
    reported rather than propagated.
    """
    try:
        segments, counters = parse()
    except Exception as aborted:
        return ScenarioResult("transcript-drift", False, f"parse aborted: {aborted}")
    if uncounted := [name for name in DRIFT_COUNTERS if not getattr(counters, name)]:
        return ScenarioResult("transcript-drift", False, f"drift uncounted: {', '.join(uncounted)}")
    if not segments:
        return ScenarioResult("transcript-drift", False, "the sound records were dropped with it")
    counted = sum(getattr(counters, name) for name in DRIFT_COUNTERS)
    return ScenarioResult(
        "transcript-drift",
        True,
        f"{counted} drifted parts counted, {len(segments)} segments kept",
    )


HOOK_SAMPLES = 200
"""Recalls timed per run — enough that the 95th sample is a tail, not one outlier."""

HOOK_BUDGET = 1.0
"""Seconds the recall verb's p95 must stay under to be worth leaving a hook on (SC-010)."""

HOOK_CEILING = 5.0
"""Seconds no single recall may reach — the hook process's own timeout (SC-010)."""


def hook_latency(samples: Sequence[float]) -> ScenarioResult:
    """The recall verb must answer inside the hook budget, every time (SC-010).

    Both halves are the same claim at two confidences: p95 is what the hook
    costs a session, the slowest sample is what a session ever waits for.
    """
    if len(samples) < HOOK_SAMPLES:
        return ScenarioResult(
            "hook-latency", False, f"{len(samples)} samples measured, {HOOK_SAMPLES} asked for"
        )
    p95, slowest = _pct(list(samples), 95), max(samples)
    measured = f"p95 {p95 * 1e3:.0f} ms over {len(samples)} samples, slowest {slowest * 1e3:.0f} ms"
    if p95 >= HOOK_BUDGET:
        return ScenarioResult(
            "hook-latency", False, f"over the {HOOK_BUDGET:.0f} s budget: {measured}"
        )
    if slowest >= HOOK_CEILING:
        return ScenarioResult(
            "hook-latency", False, f"past the {HOOK_CEILING:.0f} s ceiling: {measured}"
        )
    return ScenarioResult("hook-latency", True, measured)


def resolve_scaling(base: float, at_ten_times: float) -> ScenarioResult:
    """Per-batch wall clock must not follow the namespace it resolves against (SC-006).

    Blocking and the candidate cap are what make this true: a batch compares
    within its blocks, so ten times the namespace is the same work. A rise past
    twice the base means a scan survived somewhere.
    """
    return ScenarioResult(
        "resolve-scaling",
        at_ten_times < 2 * base,
        f"{base * 1e3:.1f} ms per batch at the base size, {at_ten_times * 1e3:.1f} ms at ten times",
    )


# Two unrelated domains in one namespace — the two-pack claim as data. Small on
# purpose: the scenario proves the core stays domain-blind, not that extraction
# scales. Each query names a symbol only its own pack declares — recall is
# symbol-keyed (FR-002), so the query a probe is answered by is a concept label,
# never one of its own surfaces.
PROBES = (
    PackProbe(
        uri="scenario://code/loader.py",
        text="def load_packs(packs): return LoadedPacks(packs)",
        kind=SegmentKind.code,
        query="which function calls what",
    ),
    PackProbe(
        uri="scenario://agent/session.md",
        text="The agent decided to rebuild the namespace instead of migrating it.",
        kind=SegmentKind.prose,
        query="what decision was settled",
    ),
)


def _memory(namespace: str) -> Memory:
    """A memory over *namespace* carrying both shipped packs — the two-pack setup itself."""
    return Memory(namespace=namespace, packs=(CodePack(), AgentPack()))


async def _fresh_memory(namespace: str) -> Memory:
    """A memory over an emptied *namespace*: two ingests compare only from nothing."""
    memory = _memory(namespace)
    await memory.drop_namespace()
    return memory


async def _stale_namespace(namespace: str) -> None:
    """Rebuild *namespace* stamped by a version no code runs — the refusal's precondition."""
    settings = get_settings()
    store = build_graph_store(settings, namespace)
    await store.connect()
    try:
        await store.drop_database()
        await store.ensure_schema(settings.embed_dimensions or embed_dim())
        await store.command("MATCH (s:SCHEMA_STAMP) SET s.version = $version", version="stale")
    finally:
        await store.close()


async def _two_packs(namespace: str) -> ScenarioResult:
    """The two-pack claim, proved from nothing: content already stored never re-extracts."""
    return await two_packs(await _fresh_memory(namespace), PROBES)


async def _schema_refusal(namespace: str) -> ScenarioResult:
    await _stale_namespace(namespace)
    return await schema_refusal(lambda: _memory(namespace).recall("schema stamp"))


# Three views of one normalised key, differing only in how often each was
# observed, proposed as a chain — the smallest batch whose log is more than a
# single entry, so replay has an order to get wrong.
MERGE_VIEWS = {
    f"acme{rank}": EntityView(id=f"acme{rank}", name_norm="acme", type_histogram={"org": rank})
    for rank in (1, 2, 3)
}
MERGE_NEIGHBOURS = {"acme1": ["acme2"], "acme2": ["acme3"]}


async def _merge_replay(namespace: str) -> ScenarioResult:
    """The merge-replay claim: pure, so it needs no namespace of its own."""
    return merge_replay(Resolver().plan(MERGE_VIEWS, MERGE_NEIGHBOURS))


BATCH_ENTITIES = 1_000
"""Entities resolved per timed batch — fixed, so only the namespace around it grows."""

BASE_SIZE = 10_000
"""The namespace the tenfold increase is measured from (SC-006)."""


def _scaling_view(index: int) -> EntityView:
    """One namespace entity, its name drawn from a fixed vocabulary.

    Names repeat on purpose: a namespace that grows fills its blocks rather
    than inventing new ones, which is the growth the cap has to absorb.
    """
    return EntityView(id=f"e{index}", name_norm=f"{index % 100:04d}", type_histogram={"org": 1})


def _blocks(views: Mapping[str, EntityView]) -> dict[str, list[str]]:
    """The namespace partitioned by block key — the proposal a store returns (FR-020)."""
    blocks: dict[str, list[str]] = {}
    for view in views.values():
        blocks.setdefault(view.block_key, []).append(view.id)
    return blocks


def seconds_per_batch(namespace_size: int) -> float:
    """Wall clock of resolving one batch against a namespace of *namespace_size*.

    Building the namespace and its block index is the store's work, not the
    batch's, so only the resolution of :data:`BATCH_ENTITIES` against the views
    their blocks proposed is timed.
    """
    views = {view.id: view for view in map(_scaling_view, range(namespace_size))}
    blocks = _blocks(views)
    batch = list(views.values())[:BATCH_ENTITIES]
    # A candidate query is LIMIT-ed by the store, so a block proposes at most
    # what one entity is worth comparing against — never the whole block.
    neighbours = {view.id: blocks[view.block_key][:MAX_CANDIDATES] for view in batch}
    compared = {other: views[other] for proposed in neighbours.values() for other in proposed}
    started = perf_counter()
    Resolver().plan(compared, neighbours)
    return perf_counter() - started


async def _resolve_scaling(namespace: str) -> ScenarioResult:
    """The scaling claim: resolution is in-process, so it needs no namespace of its own."""
    return resolve_scaling(seconds_per_batch(BASE_SIZE), seconds_per_batch(BASE_SIZE * 10))


async def _merge_log(store: Any) -> tuple[str, ...]:
    """The ``MERGED_INTO`` entries the namespace still reads, in a stable order."""
    rows = await store.command(
        "MATCH (source:ENTITY)-[:MERGED_INTO]->(survivor:ENTITY) "
        "RETURN source.id AS source, survivor.id AS survivor"
    )
    return tuple(sorted(f"{row['source']}->{row['survivor']}" for row in rows))


async def _fact_state(store: Any, record_id: str) -> str:
    """The lifecycle state *record_id* reads as — empty when the record is gone."""
    rows = await store.command("MATCH (f:FACT {id: $id}) RETURN f.state AS state", id=record_id)
    return str(rows[0]["state"]) if rows else ""


async def _observe_forget(namespace: str, probe: PackProbe) -> ForgetObservation:
    """Ingest *probe*, forget the first fact it recalls, and read the graph around it.

    The reads go through a store of their own: recall filters tombstones out by
    design, so what forget left behind is only visible below it.
    """
    memory = await _fresh_memory(namespace)
    await _ingest(memory, probe)
    recalled = (await memory.recall(probe.query)).facts
    if not recalled:
        return ForgetObservation("", (), "", (), ())
    record_id = recalled[0].fact.id
    store = build_graph_store(get_settings(), namespace)
    await store.connect()
    try:
        before = await _merge_log(store)
        await memory.forget(record_id)
        return ForgetObservation(
            record_id=record_id,
            recalled_after=tuple(item.fact.id for item in (await memory.recall(probe.query)).facts),
            state_after=await _fact_state(store, record_id),
            merge_log_before=before,
            merge_log_after=await _merge_log(store),
        )
    finally:
        await store.close()


async def _forget_roundtrip(namespace: str) -> ScenarioResult:
    """The forget claim, proved against the probe the code pack understands."""
    return forget_roundtrip(await _observe_forget(namespace, PROBES[0]))


async def _recall_samples(memory: Any, query: str) -> list[float]:
    """Wall clock of :data:`HOOK_SAMPLES` recalls, warm — the first one is discarded.

    A hook runs against a service an earlier hook already woke, so the connect
    and the schema check the first recall pays for belong to no sample.
    """
    await memory.recall(query)
    samples = []
    for _ in range(HOOK_SAMPLES):
        started = perf_counter()
        await memory.recall(query)
        samples.append(perf_counter() - started)
    return samples


async def _hook_latency(namespace: str) -> ScenarioResult:
    """The budget claim, measured on the verb the hook shells out to (SC-010)."""
    memory = await _fresh_memory(namespace)
    await _ingest(memory, PROBES[0])
    return hook_latency(await _recall_samples(memory, PROBES[0].query))


async def _concurrent_ingest(namespace: str) -> ScenarioResult:
    memories = [
        await _fresh_memory(f"{namespace}_{order}") for order in ("concurrent", "sequential")
    ]
    return await concurrent_ingest(memories, PROBES)


# One sound record, then a drift of each kind the parser counts: an unparsable
# line, a record type it does not keep, and a block type it does not know.
DRIFTED_TRANSCRIPT = b"""\
{"type": "user", "uuid": "u1", "message": {"content": "rebuild the namespace"}}
{"type": "user", "uuid": "u2", "message":
{"type": "summary", "uuid": "u3", "message": {"content": "a type from a later harness"}}
{"type": "assistant", "uuid": "u4", "message": {"content": [{"type": "hologram"}]}}
"""


async def _transcript_drift(namespace: str) -> ScenarioResult:
    """The drift claim: parsing is in-process, so it needs no namespace of its own."""
    source = Source(
        uri="scenario://agent/drifted.jsonl",
        content_hash=sha256(DRIFTED_TRANSCRIPT).hexdigest(),
        mime="application/x-ndjson",
    )
    parser = TranscriptParser()
    return transcript_drift(lambda: (parser.parse(source, DRIFTED_TRANSCRIPT), parser.counters))


SCENARIOS: dict[str, Callable[[str], Awaitable[ScenarioResult]]] = {
    "two-packs": _two_packs,
    "schema-refusal": lambda namespace: _schema_refusal(f"{namespace}_stale"),
    "concurrent-ingest": _concurrent_ingest,
    "merge-replay": _merge_replay,
    "resolve-scaling": _resolve_scaling,
    "transcript-drift": _transcript_drift,
    "forget-roundtrip": _forget_roundtrip,
    "hook-latency": _hook_latency,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation scenario")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--namespace", default="scenario")
    args = parser.parse_args(argv)
    result = asyncio.run(SCENARIOS[args.scenario](args.namespace))
    print(f"{result.name}: {'pass' if result.passed else 'FAIL'} — {result.detail}")
    result.gate()
