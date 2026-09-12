"""T030: the facade as ``Memory(packs=[...])`` with ingest, recall and forget.

No ArcadeDB and no extractor: the store is a fake and the write path is stubbed,
so what is pinned here is the facade's own behaviour — packs loaded
all-or-nothing at construction (FR-025), ingest serialised per namespace with
its queue wait reported (FR-047), recall reaching symbol recall, and forget
tombstoning rather than deleting (FR-037).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from graphknows.cli.memory import main
from graphknows.exceptions import PackConflictError
from graphknows.memory import Memory
from graphknows.models.report import IngestReport
from graphknows.models.segment import Segment
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef, PredicateRef


@dataclass(frozen=True)
class _Pack:
    """A pack claiming one concept uri, so two of them can contest it."""

    name: str
    uri: str = "demo:Person"

    def concepts(self) -> Iterable[ConceptRef]:
        yield ConceptRef(uri=self.uri, label="person", definition="A human.", pack=self.name)

    def predicates(self) -> Iterable[PredicateRef]:
        return iter(())

    def entity_labels(self) -> tuple[str, ...]:
        return ("person",)

    def prompt_addendum(self) -> str:
        return ""

    def hygiene(self) -> Callable[[str], bool]:
        return lambda surface: True

    def thresholds(self) -> Mapping[str, float]:
        return {}

    def veto(self, a: str, b: str) -> bool:
        return False


@dataclass
class _FakeStore:
    """Answers every read with nothing and records every write."""

    commands: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return []

    async def command(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.commands.append((cypher, params))
        return []


@dataclass
class _SlowPipeline:
    """A write path that takes long enough for a second ingest to queue behind it."""

    delay: float = 0.05

    async def ingest(self, source: Source, segments: Sequence[Segment]) -> IngestReport:
        await asyncio.sleep(self.delay)
        return IngestReport(source_id=source.id, segments_written=len(segments))


def _connected(namespace: str, store: _FakeStore) -> Memory:
    """A facade wired to *store*, with the lazy connect already satisfied."""
    memory = Memory(namespace=namespace)
    memory._retriever = object()
    memory._store = store
    return memory


def _source() -> Source:
    return Source(uri="file:///demo.txt", content_hash="abc", mime="text/plain")


def test_two_packs_contesting_a_symbol_load_neither() -> None:
    """Construction is where a pack conflict surfaces, not the first write (FR-025)."""
    with pytest.raises(PackConflictError):
        Memory(namespace="t030-packs", packs=[_Pack("alpha"), _Pack("beta")])


async def test_concurrent_ingest_queues_and_reports_its_wait() -> None:
    """One namespace ingests serially; the queued caller reports how long it waited."""
    memory = _connected("t030-queue", _FakeStore())
    memory._pipeline = _SlowPipeline()

    first, second = await asyncio.gather(memory.ingest(_source(), []), memory.ingest(_source(), []))

    waits = sorted(r.counters.ingest_queue_wait_ms for r in (first, second))
    assert waits[0] == 0
    assert waits[1] >= 40


async def test_recall_answers_no_evidence_when_nothing_resolves() -> None:
    """Recall reaches symbol recall and reports "nothing is known" (FR-015)."""
    memory = _connected("t030-recall", _FakeStore())

    result = await memory.recall("who is ada")

    assert result.no_evidence is True
    assert result.facts == []


async def test_forget_tombstones_rather_than_deletes() -> None:
    """Forget marks the record forgotten; nothing is removed from the graph (FR-037)."""
    store = _FakeStore()
    memory = _connected("t030-forget", store)

    await memory.forget("f1")

    assert [params["state"] for _, params in store.commands] == ["forgotten"] * 4
    assert all("DELETE" not in cypher for cypher, _ in store.commands)


def test_cli_ingest_prints_the_report(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``ingest`` command runs the file through the facade and prints its report."""
    path = tmp_path / "demo.txt"
    path.write_text("Ada wrote the first program.", encoding="utf-8")
    ingested: list[Sequence[Segment]] = []

    class _StubMemory:
        def __init__(self, namespace: str | None = None) -> None:
            self.namespace = namespace

        async def __aenter__(self) -> _StubMemory:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def ingest(self, source: Source, segments: Sequence[Segment]) -> IngestReport:
            ingested.append(segments)
            return IngestReport(source_id=source.id, segments_written=len(segments))

    monkeypatch.setattr("graphknows.cli.memory.Memory", _StubMemory)

    assert main(["ingest", str(path)]) == 0
    assert ingested[0][0].text == "Ada wrote the first program."
    assert '"segments_written": 1' in capsys.readouterr().out
