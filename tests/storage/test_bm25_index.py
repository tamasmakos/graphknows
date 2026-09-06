"""FR-020 — the BM25 candidate pool is served by a full-text index.

Three assertions, and the middle one is the requirement: the DDL list creates
the ``FULL_TEXT`` indexes; ``EXPLAIN`` on the candidate query names them rather
than a bucket scan; and the chunk ids are identical to the pre-change
``toLower(...) CONTAINS`` query's, so a scalability fix cannot hide a recall
change.

Live ArcadeDB only — ``EXPLAIN`` is the point, and a mock has no query planner.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest

from graphknows.settings import GraphKnowsSettings
from graphknows.storage import build_arcadedb_client
from graphknows.storage.arcadedb._schema import _CORE_DDL
from graphknows.storage.arcadedb.graph_store import GraphStore

# What the retriever's BM25 channel passes: stop-worded query tokens. "gina"
# appears only in `search_text` (the speaker prefix), so both halves of the OR —
# and therefore both indexes — have to answer for the pool to be unchanged.
_TERMS = ["camping", "gina"]
_SESSION = "sess-bm25"
_CHUNKS: list[dict[str, Any]] = [
    {"chunk_id": "chunk-camping", "text": "We went camping by the lake.", "speaker": "Gina"},
    {"chunk_id": "chunk-studio", "text": "The dance studio opens on Monday.", "speaker": "Gina"},
    {"chunk_id": "chunk-other", "text": "Nothing to see here.", "speaker": "Bob"},
]


def _pre_change_cypher(terms: list[str], session_id: str, state: str, top_k: int) -> str:
    """The candidate query exactly as it read before the index landed."""
    where = " OR ".join(
        f"(toLower(c.search_text) CONTAINS '{t.lower()}' OR toLower(c.text) CONTAINS '{t.lower()}')"
        for t in terms
    )
    return (
        f"MATCH (c:CHUNK) WHERE ({where}) "
        f"AND c.session_id = '{session_id}' AND c.state = '{state}' "
        f"RETURN c.chunk_id AS chunk_id, c.text AS text LIMIT {top_k}"
    )


def _explain(settings: GraphKnowsSettings, db: str, sql: str) -> str:
    """Return ArcadeDB's rendered plan for *sql*.

    Raw HTTP because the plan travels in the response's ``explain`` field, which
    ``ArcadeDBClient.query`` drops — it returns ``result``, and ``EXPLAIN``
    leaves that empty.
    """
    response = httpx.post(
        f"{settings.arcadedb_url}/api/v1/query/{db}",
        json={"language": "sql", "command": f"EXPLAIN {sql}"},
        auth=(settings.arcadedb_user, settings.arcadedb_password),
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json()["explain"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bm25_pool_is_index_served_and_unchanged(arcadedb_required: None) -> None:
    settings = GraphKnowsSettings()
    client = build_arcadedb_client(settings)
    db = f"mem_test_{uuid.uuid4().hex[:8]}"
    store = GraphStore(client=client, db=db)

    # Record the SQL the candidate pool issues, so EXPLAIN runs the real query
    # rather than a copy of it that could drift.
    issued: list[str] = []
    inner_query = client.query

    async def recording_query(
        target: str,
        command: str,
        language: str = "cypher",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if language == "sql":
            issued.append(command)
        return await inner_query(target, command, language, params)

    # 1a. The statements are in the idempotent DDL list, so every namespace —
    # including one that already exists — gets them on its next connection.
    stmts = [stmt for _lang, stmt in _CORE_DDL]
    assert "CREATE INDEX IF NOT EXISTS ON CHUNK(text) FULL_TEXT" in stmts
    assert "CREATE INDEX IF NOT EXISTS ON CHUNK(search_text) FULL_TEXT" in stmts

    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        await store.ensure_session(_SESSION, started_at="2026-09-05T00:00:00Z")
        await store.ensure_file("file-bm25", _SESSION)
        await store.write_chunks_bulk(
            [
                {
                    **c,
                    "session_id": _SESSION,
                    "file_id": "file-bm25",
                    "seq": seq,
                    "search_text": f"{c['speaker']}: {c['text']}",
                }
                for seq, c in enumerate(_CHUNKS)
            ]
        )

        # 1b. ensure_schema() actually created them on this namespace.
        indexes = await client.query(db, "SELECT FROM schema:indexes", "sql")
        full_text = {
            (row["typeName"], prop)
            for row in indexes
            if row.get("indexType") == "FULL_TEXT"
            for props in row.get("properties", [])
            for prop in props
        }
        assert ("CHUNK", "text") in full_text
        assert ("CHUNK", "search_text") in full_text

        # 2. The candidate query is served by them, not by a bucket scan.
        client.query = recording_query  # type: ignore[method-assign]
        pool = await store.full_text_search_chunks(
            _TERMS, session_id=_SESSION, top_k=10, state="raw"
        )
        client.query = inner_query  # type: ignore[method-assign]
        assert len(issued) == 1, f"expected one SQL candidate query, got {issued}"
        plan = _explain(settings, db, issued[0])
        assert "FETCH FROM INDEX CHUNK[text]" in plan, plan
        assert "FETCH FROM INDEX CHUNK[search_text]" in plan, plan
        assert "SCAN WITH FILTER BUCKET" not in plan, plan

        # 3. Same chunks as before the change — a plan change, not a recall change.
        before = await store.query(_pre_change_cypher(_TERMS, _SESSION, "raw", 10))
        assert {r["chunk_id"] for r in before} == {"chunk-camping", "chunk-studio"}
        assert {c["chunk_id"] for c in pool} == {r["chunk_id"] for r in before}
    finally:
        await store.drop_database()
        await store.close()
