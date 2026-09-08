"""A second flush must continue the dialog file's chunk sequence, not restart it.

``CHUNK.seq`` is a position within a FILE, not a per-batch counter:
``neighbor_chunks`` expands a retrieved chunk to its siblings by walking
``(file_id, seq ± radius)``. Every flush of a session ingests into the same
``dialog-<session>`` file, so a flush that numbers its turns from 0 again mints
a second chunk at seq 0, seq 1, ... — and the sibling walk then picks between
two chunks sharing one position. This pins the read that prevents it
(``GraphStore.max_chunk_seq``) against a live ArcadeDB, and the wiring in
``Memory._ingest_buffered_turns`` that turns it into ``turn_index_offset``.

The extraction stack is stubbed out — GLiNER and embeddings are what the TURN
buffer exists to defer, and none of it is on the seam under test. The stub
writes its chunks at ``turn_index_offset + i``, which is exactly what
``STMService.ingest_messages`` does with the offset it is handed.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from graphknows.memory import Memory
from graphknows.models.ingest import IngestResult
from graphknows.models.scope import MemoryScope
from graphknows.settings import GraphKnowsSettings, MemoryMode
from graphknows.storage import build_arcadedb_client
from graphknows.storage.arcadedb.graph_store import GraphStore


class _RecordingService:
    """Stand-in for ``STMService``: writes one CHUNK per message at the offset."""

    def __init__(self, store: GraphStore, offsets: list[int]) -> None:
        self._store = store
        self._offsets = offsets

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ingest_messages(
        self, messages: list[Any], options: Any, infer_flags: list[bool] | None = None
    ) -> IngestResult:
        self._offsets.append(options.turn_index_offset)
        session_id = options.scope.to_session_id()
        file_id = options.scope.to_dialog_id()
        await self._store.write_chunks_bulk(
            [
                {
                    "chunk_id": uuid.uuid4().hex[:16],
                    "session_id": session_id,
                    "file_id": file_id,
                    "text": m.content,
                    "seq": options.turn_index_offset + i,
                }
                for i, m in enumerate(messages)
            ]
        )
        return IngestResult(
            session_id=session_id,
            file_id=file_id,
            title="",
            chunk_count=len(messages),
        )


def _raw_turns(texts: list[str]) -> list[dict[str, Any]]:
    """Buffered TURN rows as ``list_turns`` returns them: newest first."""
    return [
        {"role": "user", "text": t, "ts": "", "speaker": "", "infer": True} for t in reversed(texts)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_flush_continues_chunk_seq(
    arcadedb_required: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two drains of the same session leave every chunk on its own seq."""
    import graphknows.ingestion as ingestion

    settings = GraphKnowsSettings()
    client = build_arcadedb_client(settings)
    store = GraphStore(client=client, db=f"mem_test_{uuid.uuid4().hex[:8]}")
    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        session_id = "sess-flush-seq"
        file_id = MemoryScope(run_id=session_id).to_dialog_id()
        await store.ensure_session(session_id, started_at="2026-09-06T00:00:00Z")
        await store.ensure_file(file_id, session_id)

        # An untouched file has no sequence to continue.
        assert await store.max_chunk_seq(file_id) is None

        offsets: list[int] = []
        monkeypatch.setattr(
            ingestion, "build_stm_service", lambda *a, **k: _RecordingService(store, offsets)
        )
        memory = Memory(settings=settings, mode=MemoryMode.llm_free)

        await memory._ingest_buffered_turns(store, session_id, _raw_turns(["one", "two"]))
        await memory._ingest_buffered_turns(store, session_id, _raw_turns(["three", "four"]))

        assert offsets == [0, 2], "second flush restarted the sequence"
        assert await store.max_chunk_seq(file_id) == 3

        rows = await store.query(
            "MATCH (c:CHUNK)-[:PART_OF]->(f:FILE {file_id: $fid}) RETURN c.seq AS seq",
            fid=file_id,
        )
        seqs = sorted(int(r["seq"]) for r in rows)
        assert seqs == [0, 1, 2, 3], "two chunks share one (file_id, seq) position"
    finally:
        try:
            await store.drop_database()
        finally:
            await store.close()
