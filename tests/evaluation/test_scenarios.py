"""The three cutover scenarios hold, and a failing one gates the run (FR-042)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.scenarios import (
    PROBES,
    SCENARIOS,
    PackProbe,
    ScenarioResult,
    concurrent_ingest,
    schema_refusal,
    two_packs,
)
from graphknows.exceptions import SchemaVersionMismatchError
from graphknows.models import SegmentKind

pytestmark = pytest.mark.unit


def report(facts: int = 1, entities: int = 1) -> SimpleNamespace:
    """An ingest report shaped like the fields the scenarios read."""
    return SimpleNamespace(facts_written=facts, entities_touched=entities)


class FakeMemory:
    """A memory that records what was ingested and answers the queries it was given."""

    def __init__(self, answers: dict[str, int] | None = None) -> None:
        self.answers = answers or {}
        self.ingested: list[str] = []

    async def ingest(self, source, segments) -> SimpleNamespace:
        self.ingested.append(source.uri)
        return report(len(segments), len(segments))

    async def recall(self, query: str) -> SimpleNamespace:
        return SimpleNamespace(facts=[object()] * self.answers.get(query, 1))


class TestScenarioResult:
    def test_a_failure_exits_non_zero_with_its_reason(self) -> None:
        with pytest.raises(SystemExit, match="no facts"):
            ScenarioResult("two-packs", False, "no facts recalled for: x").gate()

    def test_a_pass_is_a_no_op(self) -> None:
        assert ScenarioResult("two-packs", True, "every probe recalled").gate() is None


class TestPackProbe:
    def test_the_segment_is_cut_from_its_own_source(self) -> None:
        probe = PackProbe(uri="scenario://a", text="hello", kind=SegmentKind.prose, query="hi")
        (segment,) = probe.segments
        assert segment.source_id == probe.source.id
        assert segment.byte_range == (0, 5)

    def test_the_shipped_probes_cover_two_unrelated_domains(self) -> None:
        assert {probe.kind for probe in PROBES} == {SegmentKind.code, SegmentKind.prose}


class TestTwoPacks:
    async def test_every_probe_ingests_and_recalls(self) -> None:
        memory = FakeMemory()
        result = await two_packs(memory, PROBES)
        assert memory.ingested == [probe.uri for probe in PROBES]
        assert result.passed

    async def test_a_probe_that_recalls_nothing_fails_and_is_named(self) -> None:
        silent = PROBES[1].query
        result = await two_packs(FakeMemory({silent: 0}), PROBES)
        assert not result.passed
        assert silent in result.detail


class TestSchemaRefusal:
    async def test_a_mismatch_passes_and_names_both_versions(self) -> None:
        async def open_namespace() -> None:
            raise SchemaVersionMismatchError("scratch", "mem_scratch", "v1", "v2")

        result = await schema_refusal(open_namespace)
        assert result.passed
        assert "v1" in result.detail and "v2" in result.detail

    async def test_an_unstamped_namespace_still_names_the_running_version(self) -> None:
        async def open_namespace() -> None:
            raise SchemaVersionMismatchError("scratch", "mem_scratch", None, "v2")

        assert (await schema_refusal(open_namespace)).detail == "refused: recorded none, running v2"

    async def test_opening_it_anyway_fails_the_scenario(self) -> None:
        async def open_namespace() -> None:
            return None

        assert not (await schema_refusal(open_namespace)).passed


class TestConcurrentIngest:
    async def test_the_same_writes_in_either_order_pass(self) -> None:
        result = await concurrent_ingest((FakeMemory(), FakeMemory()), PROBES)
        assert result.passed

    async def test_a_divergent_concurrent_graph_fails(self) -> None:
        class LosesAWrite(FakeMemory):
            async def ingest(self, source, segments) -> SimpleNamespace:
                await super().ingest(source, segments)
                return report(0, 0)

        result = await concurrent_ingest((LosesAWrite(), FakeMemory()), PROBES)
        assert not result.passed
        assert "concurrent wrote (0, 0)" in result.detail

    async def test_a_failed_concurrent_ingest_is_reported_not_raised(self) -> None:
        class Refuses(FakeMemory):
            async def ingest(self, source, segments) -> SimpleNamespace:
                raise RuntimeError("lock timeout")

        result = await concurrent_ingest((Refuses(), FakeMemory()), PROBES)
        assert not result.passed
        assert "lock timeout" in result.detail


def test_the_cli_offers_every_gating_scenario() -> None:
    assert set(SCENARIOS) == {
        "two-packs",
        "schema-refusal",
        "concurrent-ingest",
        "merge-replay",
        "resolve-scaling",
        "transcript-drift",
        "forget-roundtrip",
        "hook-latency",
    }
