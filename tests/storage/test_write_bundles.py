"""FR-028/SC-018: the two flat parameter lists are bundle dataclasses now.

The audit named exactly two signatures: the 17-parameter ``write_relation``
(``storage/arcadedb/graph_store.py``) and the 16-parameter ``_write_extracted``
(``ingestion/stm/ingest.py``). Both now take one frozen dataclass — mirroring
the existing ``OntologyTerm`` pattern — in place of their parameter list. This
file pins the shape (one bundle parameter, frozen) and that each still does
its one job when handed a bundle.
"""

from __future__ import annotations

import dataclasses
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import ExtractedChunk, _STMIngestHandler
from graphknows.storage.arcadedb.graph_store import GraphStore, RelationWrite


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return client


# ---------------------------------------------------------------------------
# The bundles are frozen dataclasses, mirroring OntologyTerm.
# ---------------------------------------------------------------------------


def test_relation_write_is_a_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(RelationWrite)
    write = RelationWrite(src_name="a", predicate="LIKES", dst_name="b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        write.predicate = "LIVES_IN"  # type: ignore[misc]


def test_extracted_chunk_is_a_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(ExtractedChunk)
    chunk = ExtractedChunk(
        session_id="s1",
        file_id="f1",
        chunk_id="c1",
        text="hi",
        seq=0,
        kind="text",
        speaker="",
        ts="",
        embedding=[0.1],
        extraction=ExtractionResult(),
        temporal=[],
        memory_class="raw_evidence",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.chunk_id = "c2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Each signature takes its bundle in place of its parameter list.
# ---------------------------------------------------------------------------


def test_write_relation_takes_one_bundle_parameter_not_seventeen() -> None:
    params = list(inspect.signature(GraphStore.write_relation).parameters.values())
    assert [p.name for p in params] == ["self", "write"]
    assert params[1].annotation in ("RelationWrite", RelationWrite)


def test_write_extracted_takes_one_bundle_parameter_not_sixteen() -> None:
    params = list(inspect.signature(_STMIngestHandler._write_extracted).parameters.values())
    assert [p.name for p in params] == ["self", "chunk"]
    assert params[1].annotation in ("ExtractedChunk", ExtractedChunk)


# ---------------------------------------------------------------------------
# Both still do their one job when handed a bundle.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_relation_merges_the_edge_from_its_bundle() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_relation(
        RelationWrite(
            src_name="Melanie",
            predicate="LIKES",
            dst_name="Camping",
            fact="Melanie likes camping",
            chunk_id="chunk-1",
            confidence=0.8,
        )
    )

    merge_call = next(c for c in client.command.await_args_list if "MERGE (a)-[r:REL" in c.args[1])
    assert merge_call.kwargs["params"]["src_nn"] == "melanie"
    assert merge_call.kwargs["params"]["dst_nn"] == "camping"
    assert merge_call.kwargs["params"]["predicate"] == "LIKES"


@pytest.mark.asyncio
async def test_write_extracted_writes_the_chunk_from_its_bundle() -> None:
    store = MagicMock()
    for name in (
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_temporal",
        "write_frames",
        "write_frame_instances",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    handler = _STMIngestHandler(store, extractor=None, embed_fn=None)

    entities = [{"name": "Mel", "type": "PERSON", "confidence": 0.9}]
    entity_count = await handler._write_extracted(
        ExtractedChunk(
            session_id="s1",
            file_id="f1",
            chunk_id="c1",
            text="Mel went running.",
            seq=0,
            kind="text",
            speaker="Mel",
            ts="",
            embedding=[0.1],
            extraction=ExtractionResult(entities=entities),
            temporal=[],
            memory_class="raw_evidence",
        )
    )

    assert entity_count == 1
    store.write_chunk.assert_awaited_once()
    written_chunk = store.write_chunk.await_args.args[0]
    assert written_chunk["chunk_id"] == "c1"
    assert written_chunk["session_id"] == "s1"
