"""T025/FR-007: content already stored is dropped before any model runs.

Mocked client, no ArcadeDB: what is pinned is that a matching ``content_hash``
short-circuits ingest — no writes, no extractor call — and says so in the
``sources_deduplicated`` counter (SC-008).
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

from graphknows.ingestion.pipeline import IngestPipeline
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source
from graphknows.storage.arcadedb._base import ArcadeStoreBase


def _no_model(texts: Sequence[str]) -> list[list[float]]:
    """A stand-in encoder, so no test here loads an embedding model."""
    return [[1.0] for _ in texts]


class _FailingExtractor:
    """Any call is a defect: dedup happens before the models run."""

    name = "never"
    version = "1"

    def extract(self, segment: Segment, pack: object) -> tuple[list[Mention], list[Fact]]:
        raise AssertionError("extractor ran on a deduplicated source")


def _store(known: bool) -> ArcadeStoreBase:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[{"id": "seen"}] if known else [])
    return ArcadeStoreBase(client, "db")


def _source() -> Source:
    return Source(uri="file:///notes.md", content_hash="abc", mime="text/markdown")


def _segment(source: Source) -> Segment:
    return Segment(
        source_id=source.id,
        text="Ada designed the Engine.",
        kind=SegmentKind.prose,
        byte_range=(0, 24),
    )


async def test_known_content_is_counted_and_nothing_is_written() -> None:
    store = _store(known=True)
    source = _source()

    report = await IngestPipeline(store, _FailingExtractor(), [], embed=_no_model).ingest(
        source, [_segment(source)]
    )

    assert report.counters.sources_deduplicated == 1
    assert (report.segments_written, report.facts_written) == (0, 0)
    store.client.command.assert_not_awaited()


async def test_unknown_content_is_written_and_not_counted() -> None:
    store = _store(known=False)
    source = _source()

    report = await IngestPipeline(store, _FailingExtractor(), [], embed=_no_model).ingest(
        source, [_segment(source)]
    )

    assert report.counters.sources_deduplicated == 0
    assert report.segments_written == 1
    statements = [call.args[1] for call in store.client.command.await_args_list]
    assert any("UPDATE SOURCE SET" in statement for statement in statements)


async def test_the_dedup_lookup_is_keyed_on_content_hash() -> None:
    store = _store(known=True)
    source = _source()

    await IngestPipeline(store, _FailingExtractor(), [], embed=_no_model).ingest(source, [])

    cypher, params = store.client.query.await_args.args[1], store.client.query.await_args.kwargs
    assert "content_hash: $content_hash" in cypher
    assert params["params"] == {"content_hash": source.content_hash}
