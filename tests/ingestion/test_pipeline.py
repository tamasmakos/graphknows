"""T024/FR-002: the Source -> Segment -> Mention/Fact write path and its labelling.

Mocked client, no ArcadeDB: what is pinned is the order of the writes and the
two symbol edges bottom-up labelling owes top-down activation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.pipeline import IngestPipeline
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.storage.arcadedb._base import ArcadeStoreBase


@dataclass(frozen=True)
class _StubPack:
    """A pack naming one concept, under the label its extractor answers with."""

    name: str = "demo"

    def concepts(self) -> Iterable[ConceptRef]:
        yield ConceptRef(
            uri="demo:Person",
            label="person",
            definition="A human being.",
            pack=self.name,
        )

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


class _StubExtractor:
    """Answers with one typed mention, one untyped one, and a fact between them."""

    name = "stub"
    version = "1"

    def extract(self, segment: Segment, pack: object) -> tuple[list[Mention], list[Fact]]:
        ada = Mention(
            segment_id=segment.id,
            entity_id="e1",
            surface="Ada",
            span=(0, 3),
            label="PERSON",
            confidence=0.9,
        )
        engine = Mention(segment_id=segment.id, entity_id="e2", surface="the Engine", span=(13, 23))
        fact = Fact(id="f1", subject="e1", predicate="demo:built", object="e2")
        return [ada, engine], [fact]


@pytest.fixture
def store() -> ArcadeStoreBase:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return ArcadeStoreBase(client, "db")


def _statements(store: ArcadeStoreBase) -> list[str]:
    return [call.args[1] for call in store.client.command.await_args_list]


def _params(store: ArcadeStoreBase, fragment: str) -> dict:
    """Parameters of the one command whose text contains *fragment*."""
    calls = [c for c in store.client.command.await_args_list if fragment in c.args[1]]
    assert len(calls) == 1, f"{fragment!r} issued {len(calls)} times"
    return calls[0].kwargs["params"]


def _segment(source: Source) -> Segment:
    return Segment(
        source_id=source.id,
        text="Ada designed the Engine.",
        kind=SegmentKind.prose,
        byte_range=(0, 24),
    )


def _source() -> Source:
    return Source(uri="file:///notes.md", content_hash="abc", mime="text/markdown")


async def test_a_source_is_written_as_segments_mentions_and_facts(
    store: ArcadeStoreBase,
) -> None:
    source, pipeline = _source(), IngestPipeline(store, _StubExtractor(), [_StubPack()])
    segment = _segment(source)

    report = await pipeline.ingest(source, [segment])

    text = "\n".join(_statements(store))
    assert "MERGE (s:SOURCE {id: $id})" in text
    assert "MERGE (g:SEGMENT {id: $id})" in text
    assert "MERGE (e:ENTITY {id: $id})" in text
    assert "MERGE (g)-[m:MENTIONS" in text
    assert "MERGE (f:FACT {id: $id})" in text
    assert (report.source_id, report.segments_written) == (source.id, 1)
    assert (report.facts_written, report.entities_touched) == (1, 2)


async def test_a_fact_is_written_against_the_segment_that_evidences_it(
    store: ArcadeStoreBase,
) -> None:
    source = _source()
    segment = _segment(source)

    await IngestPipeline(store, _StubExtractor(), [_StubPack()]).ingest(source, [segment])

    evidence = _params(store, "ASSERTED_IN")
    assert evidence["segment_id"] == segment.id
    assert evidence["span"] == "0:24"


async def test_a_labelled_mention_evokes_a_concept_and_types_its_entity(
    store: ArcadeStoreBase,
) -> None:
    source = _source()
    segment = _segment(source)

    await IngestPipeline(store, _StubExtractor(), [_StubPack()]).ingest(source, [segment])

    instance_of = _params(store, "INSTANCE_OF")
    assert (instance_of["entity_id"], instance_of["uri"]) == ("e1", "demo:Person")
    assert (instance_of["score"], instance_of["pack"]) == (0.9, "demo")
    evokes = _params(store, "EVOKES")
    assert (evokes["segment_id"], evokes["uri"]) == (segment.id, "demo:Person")
    assert evokes["confidence"] == 0.9


async def test_a_mention_no_pack_labels_is_written_but_not_typed(
    store: ArcadeStoreBase,
) -> None:
    source = _source()

    await IngestPipeline(store, _StubExtractor(), [_StubPack()]).ingest(source, [_segment(source)])

    statements = _statements(store)
    assert sum("MERGE (e:ENTITY {id: $id})" in s for s in statements) == 2
    assert sum("INSTANCE_OF" in s for s in statements) == 1
