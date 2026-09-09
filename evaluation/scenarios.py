"""The cutover scenarios (FR-042): ``python -m evaluation scenario <name>``.

Three of them gate this slice's merge, and each one is a claim about the core
rather than a benchmark number:

* ``two-packs`` — two unrelated packs share one namespace and both recall facts
  with evidence, so the core knows no domain (SC-002).
* ``schema-refusal`` — a namespace stamped by another schema refuses to open,
  naming both versions, instead of migrating it (SC-013).
* ``concurrent-ingest`` — ingesting concurrently lands the same graph as
  ingesting in sequence, and neither ingest fails (SC-015).

Each scenario takes what it drives and returns a :class:`ScenarioResult`; the
run outcome is :meth:`ScenarioResult.gate`, the same shape the dead-weight check
uses, so a failed scenario exits non-zero with its reason.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from graphknows.exceptions import SchemaVersionMismatchError
from graphknows.memory import Memory
from graphknows.models import Segment, SegmentKind, Source
from graphknows.packs.agent import AgentPack
from graphknows.packs.code import CodePack


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


# Two unrelated domains in one namespace — the two-pack claim as data. Small on
# purpose: the scenario proves the core stays domain-blind, not that extraction
# scales.
PROBES = (
    PackProbe(
        uri="scenario://code/loader.py",
        text="def load_packs(packs): return LoadedPacks(packs)",
        kind=SegmentKind.code,
        query="load_packs",
    ),
    PackProbe(
        uri="scenario://agent/session.md",
        text="The agent decided to rebuild the namespace instead of migrating it.",
        kind=SegmentKind.prose,
        query="rebuild the namespace",
    ),
)


def _memory(namespace: str) -> Memory:
    """A memory over *namespace* carrying both shipped packs — the two-pack setup itself."""
    return Memory(namespace=namespace, packs=(CodePack(), AgentPack()))


SCENARIOS: dict[str, Callable[[str], Awaitable[ScenarioResult]]] = {
    "two-packs": lambda namespace: two_packs(_memory(namespace), PROBES),
    "schema-refusal": lambda namespace: schema_refusal(
        lambda: _memory(namespace).recall("schema stamp")
    ),
    "concurrent-ingest": lambda namespace: concurrent_ingest(
        (_memory(namespace), _memory(f"{namespace}_sequential")), PROBES
    ),
}


# The interim identity state, stated in the cutover report (FR-048): the merge
# planes stay declared but unwritten until the identity story lands.
EXACT_KEY_ONLY = "identity: exact-key only — no MERGED_INTO and no SAME_AS edge is written (FR-048)"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation scenario")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--namespace", default="scenario")
    args = parser.parse_args(argv)
    result = asyncio.run(SCENARIOS[args.scenario](args.namespace))
    print(f"{result.name}: {'pass' if result.passed else 'FAIL'} — {result.detail}")
    print(EXACT_KEY_ONLY)
    result.gate()
