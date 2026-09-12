"""T026/FR-012: the currency rule a functional predicate imposes on its facts.

Mocked client, no ArcadeDB: what is pinned is which of two competing facts is
retired, and that facts anchored at one instant contradict instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.pipeline import IngestPipeline
from graphknows.models.fact import Fact, Mention, Validity
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.storage.arcadedb._base import ArcadeStoreBase


def _no_model(texts: Sequence[str]) -> list[list[float]]:
    """A stand-in encoder, so no test here loads an embedding model."""
    return [[1.0] for _ in texts]


EARLIER = datetime(2024, 1, 1, tzinfo=UTC)
LATER = datetime(2025, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _StubPack:
    """A pack naming one predicate, functional or not as the test needs."""

    functional: bool = True
    name: str = "demo"

    def concepts(self) -> Iterable[ConceptRef]:
        return iter(())

    def predicates(self) -> Iterable[PredicateRef]:
        yield PredicateRef(
            id="demo:lives_in",
            label="lives in",
            definition="Where a person currently lives.",
            canonical="lives_in",
            functional=self.functional,
            pack=self.name,
        )

    def entity_labels(self) -> tuple[str, ...]:
        return ()

    def prompt_addendum(self) -> str:
        return ""

    def hygiene(self) -> Callable[[str], bool]:
        return lambda surface: True

    def thresholds(self) -> Mapping[str, float]:
        return {}


@dataclass(frozen=True)
class _StubExtractor:
    """Answers with one fact on the pack's predicate, anchored where asked."""

    valid_from: datetime | None = None
    name: str = "stub"
    version: str = "1"

    def extract(self, segment: Segment, pack: object) -> tuple[list[Mention], list[Fact]]:
        fact = Fact(
            id="new",
            subject="e1",
            predicate="demo:lives_in",
            object="e2",
            validity=Validity(valid_from=self.valid_from),
        )
        return [], [fact]


@pytest.fixture
def store() -> ArcadeStoreBase:
    """A store whose only stored fact is an incumbent anchored at ``EARLIER``."""
    client = MagicMock()
    client.command = AsyncMock(return_value=[])

    async def _query(db: str, cypher: str, params: dict | None = None) -> list[dict]:
        if "f.is_current = true" not in cypher:
            return []
        return [{"id": "old", "valid_from": EARLIER.isoformat()}]

    client.query = AsyncMock(side_effect=_query)
    return ArcadeStoreBase(client, "db")


def _params(store: ArcadeStoreBase, fragment: str) -> dict:
    """Parameters of the one command whose text contains *fragment*."""
    calls = [c for c in store.client.command.await_args_list if fragment in c.args[1]]
    assert len(calls) == 1, f"{fragment!r} issued {len(calls)} times"
    return calls[0].kwargs["params"]


def _issued(store: ArcadeStoreBase, fragment: str) -> bool:
    return any(fragment in call.args[1] for call in store.client.command.await_args_list)


async def _ingest(store: ArcadeStoreBase, extractor: _StubExtractor, pack: _StubPack) -> None:
    source = Source(uri="file:///notes.md", content_hash="abc", mime="text/markdown")
    segment = Segment(
        source_id=source.id,
        text="Ada lives in London.",
        kind=SegmentKind.prose,
        byte_range=(0, 20),
        observed_at=LATER,
    )
    await IngestPipeline(store, extractor, [pack], embed=_no_model).ingest(source, [segment])


async def test_a_newer_fact_supersedes_the_current_one(store: ArcadeStoreBase) -> None:
    await _ingest(store, _StubExtractor(valid_from=LATER), _StubPack())

    assert _params(store, "SUPERSEDES") == {"winner": "new", "loser": "old"}
    assert not _issued(store, "CONTRADICTS")


async def test_an_older_fact_is_superseded_by_the_current_one(store: ArcadeStoreBase) -> None:
    await _ingest(store, _StubExtractor(valid_from=datetime(2023, 1, 1, tzinfo=UTC)), _StubPack())

    assert _params(store, "SUPERSEDES") == {"winner": "old", "loser": "new"}


async def test_the_superseded_fact_is_retained_and_marked_not_current(
    store: ArcadeStoreBase,
) -> None:
    await _ingest(store, _StubExtractor(valid_from=LATER), _StubPack())

    calls = [c for c in store.client.command.await_args_list if "SUPERSEDES" in c.args[1]]
    assert "SET l.is_current = false" in calls[0].args[1]


async def test_facts_anchored_at_the_same_instant_contradict(store: ArcadeStoreBase) -> None:
    await _ingest(store, _StubExtractor(valid_from=EARLIER), _StubPack())

    assert _params(store, "CONTRADICTS") == {"id": "new", "other": "old"}
    assert not _issued(store, "SUPERSEDES")


async def test_a_fact_falls_back_to_its_segments_observation_time(
    store: ArcadeStoreBase,
) -> None:
    await _ingest(store, _StubExtractor(), _StubPack())

    assert _params(store, "SUPERSEDES") == {"winner": "new", "loser": "old"}


async def test_a_multi_valued_predicate_never_retires_its_other_objects(
    store: ArcadeStoreBase,
) -> None:
    await _ingest(store, _StubExtractor(valid_from=LATER), _StubPack(functional=False))

    assert not _issued(store, "SUPERSEDES")
    assert not _issued(store, "CONTRADICTS")
