"""Unit + integration tests for the single-database GraphStore (golden layer).

Unit tests mock ArcadeDBClient (same style as tests/memory/stores) and assert
the SQL/Cypher shapes the Phase-1b/1c rewiring will depend on. The integration
test runs only when ArcadeDB is reachable and exercises the full write→read
round trip on a scratch ``mem_test_<uuid>`` database.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import (
    _CHUNK_BATCH,
    _CORE_DDL,
    GraphStore,
)
from graphknows.storage.namespace import db_name


def _mock_client(query_rows: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=query_rows or [])
    client.create_database = AsyncMock()
    client.drop_database = AsyncMock()
    return client


def _command_strings(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


# ---------------------------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------------------------


def test_db_name_mapping() -> None:
    assert db_name("") == "mem"
    assert db_name("  ") == "mem"
    assert db_name("acme") == "mem_acme"
    assert db_name("Acme Corp!") == "mem_acme_corp"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------


def test_core_ddl_contains_golden_types_and_not_dead_ones() -> None:
    stmts = [stmt for _, stmt in _CORE_DDL]
    type_stmts = [s for s in stmts if s.startswith(("CREATE VERTEX TYPE", "CREATE EDGE TYPE"))]

    def created(kind: str, name: str) -> bool:
        return any(s.startswith(f"CREATE {kind} TYPE {name} ") for s in type_stmts)

    # Golden vertices (FRAME_INSTANCE is the role-filled frame layer — WS D3).
    # COMMUNITY was retired with compute_community_quality: ArcadeDB's
    # modularityScore is untrustworthy and the topic stage's Newman modularity
    # (graphknows.topics.metrics.newman_modularity) is the real number now.
    for vertex in (
        "SESSION",
        "TURN",
        "FILE",
        "CHUNK",
        "ENTITY",
        "TEMPORAL",
        "TOPIC",
        "FRAME",
        "FRAME_INSTANCE",
    ):
        assert created("VERTEX", vertex), f"missing vertex {vertex}"
    # Golden edges (PLAYS_ROLE / FRAME_EVOKED_IN are the FRAME_INSTANCE layer edges).
    for edge in (
        "PART_OF",
        "IN_SESSION",
        "MENTIONS",
        "MENTIONS_DATE",
        "IN_TOPIC",
        "EVOKED_BY",
        "PLAYS_ROLE",
        "FRAME_EVOKED_IN",
    ):
        assert created("EDGE", edge), f"missing edge {edge}"
    # IN_TOPIC (ENTITY -> TOPIC) carries the cluster-membership pagerank.
    assert "CREATE PROPERTY IN_TOPIC.pagerank IF NOT EXISTS DOUBLE" in stmts
    # Dead schema must NOT be recreated. TURN used to sit in this list: the
    # pre-golden-layer TURN was a dialog-turn model (DIALOG-[:HAS_TURN]->TURN
    # with NEXT/MENTIONS edges) and that one stays dead — HAS_TURN below still
    # guards it. The TURN asserted above is a different vertex: a flat
    # short-term buffer keyed by session_id, linked with IN_SESSION, drained by
    # flush(). Same name, unrelated shape.
    for vertex in ("DIALOG", "SEGMENT", "DOC", "SESSIONENTITY", "COMMUNITY"):
        assert not created("VERTEX", vertex), f"dead vertex {vertex} present"
    # HAS_TOPIC and TOPIC.embedding were write-only once the topic read path
    # went: nothing traversed CHUNK->TOPIC and nothing vector-searched TOPIC.
    assert not created("EDGE", "HAS_TOPIC"), "write-only HAS_TOPIC edge recreated"
    assert not any("TOPIC.embedding" in s for s in stmts), "write-only TOPIC.embedding recreated"
    assert not any("ON TOPIC(embedding)" in s for s in stmts), "dead TOPIC vector index recreated"
    for edge in (
        "HAS_CHUNK",
        "HAS_TURN",
        "HAS_ENTITY",
        "NEXT",
        "CO_OCCURS",
        "APPEARED_IN",
        "CONTAINS_DOC",
        "HAS_FILE",
        "IN_COMMUNITY",
    ):
        assert not created("EDGE", edge), f"dead edge {edge} present"

    # Golden naming conventions: unique keys and no legacy properties.
    joined = " | ".join(stmts)
    assert "CREATE INDEX ON ENTITY(name_norm) UNIQUE" in joined
    assert "CREATE INDEX ON CHUNK(chunk_id) UNIQUE" in joined
    assert "CREATE INDEX ON TEMPORAL(date) UNIQUE" in joined
    assert "FRAME.chunk_id" not in joined  # evidence via EVOKED_BY edge
    assert "ENTITY.label" not in joined  # `type`, not `label`
    # Lifecycle state is NOTUNIQUE-indexed on CHUNK and ENTITY.
    assert "CREATE INDEX ON CHUNK(state) NOTUNIQUE" in joined
    assert "CREATE INDEX ON ENTITY(state) NOTUNIQUE" in joined


@pytest.mark.asyncio
async def test_ensure_schema_injects_dims_and_swallows_already_exists() -> None:
    client = _mock_client()
    client.command = AsyncMock(side_effect=Exception("Type already exists"))
    store = GraphStore(client=client, db="mem")
    await store.ensure_schema(dims=8)  # must not raise
    client.create_database.assert_awaited_once_with("mem")
    vector_stmts = [s for s in _command_strings(client) if "LSM_VECTOR" in s]
    assert vector_stmts, "no vector-index DDL executed"
    assert all("dimensions:8" in s for s in vector_stmts)
    assert not any("__VEC_DIM__" in s for s in _command_strings(client))


@pytest.mark.asyncio
async def test_ensure_schema_raises_on_real_error() -> None:
    client = _mock_client()
    client.command = AsyncMock(side_effect=Exception("connection refused"))
    store = GraphStore(client=client)
    with pytest.raises(Exception, match="connection refused"):
        await store.ensure_schema(dims=8)


# ---------------------------------------------------------------------------
# Entity writes — name_norm casefolding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_entities_bulk_casefolds_name_norm() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_entities_bulk(
        session_id="s1",
        chunk_id="c1",
        entities=[
            {"name": "Melanie", "type": "PERSON", "confidence": 0.9},
            {"name": "  STRASSE  Café ", "type": "PLACE"},
        ],
    )
    assert client.command.await_count == 2
    node_call = client.command.await_args_list[0]
    rows = node_call.kwargs["params"]["rows"]
    assert rows[0]["nn"] == "melanie"
    assert rows[0]["name"] == "Melanie"
    # casefold (not lower) + whitespace collapse
    assert rows[1]["nn"] == "strasse café"
    assert "MERGE (e:ENTITY {name_norm: row.nn})" in node_call.args[1]
    # MENTIONS pass carries confidence
    edge_call = client.command.await_args_list[1]
    assert "MERGE (c)-[m:MENTIONS]->(e)" in edge_call.args[1]
    assert "m.confidence" in edge_call.args[1]


@pytest.mark.asyncio
async def test_write_entities_bulk_persists_alias_candidates() -> None:
    """Overlap resolution's discarded surfaces must reach ENTITY.aliases.

    The extractor resolves overlapping spans to one entity and records the
    surfaces it dropped ("offers", "new" for "new offers") as alias candidates.
    Those are the ONLY record that the shorter surface was ever seen, because
    resolution stops them reaching the writer as entities in their own right —
    so if the write drops them, a later chunk saying just "offers" can never
    match the node. Computed-then-discarded is the failure mode CLAUDE.md names.

    Asserts on the VALUE reaching the query, not that a write happened.
    """
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_entities_bulk(
        session_id="s1",
        chunk_id="c1",
        entities=[
            {
                "name": "new offers",
                "type": "PRODUCT",
                "confidence": 0.8,
                "aliases": ["offers", "new"],
            },
        ],
    )
    node_call = client.command.await_args_list[0]
    rows = node_call.kwargs["params"]["rows"]
    assert rows[0]["aliases"] == ["offers", "new"], rows[0]
    # The MERGE must union the incoming candidates in, not just the surface form,
    # and must be idempotent: re-ingesting a chunk cannot grow the alias list.
    query = node_call.args[1]
    assert "row.aliases" in query
    assert "[a IN row.aliases WHERE NOT a IN coalesce(e.aliases, [])]" in query


@pytest.mark.asyncio
async def test_write_entities_bulk_alias_candidates_default_empty() -> None:
    """An entity with no discarded overlap carries no aliases — not a None."""
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_entities_bulk(
        session_id="s1", chunk_id="c1", entities=[{"name": "Gina", "type": "PERSON"}]
    )
    rows = client.command.await_args_list[0].kwargs["params"]["rows"]
    assert rows[0]["aliases"] == []


@pytest.mark.asyncio
async def test_write_entities_bulk_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_entities_bulk(session_id="s1", chunk_id="c1", entities=[])
    client.command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Topic writes — IN_TOPIC (entity membership) is the whole persisted layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_topics_emits_in_topic_edges_with_pagerank_on_name_norm() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    written = await store.write_topics(
        [
            {
                "topic_id": "topic:abc:1",
                "summary": "summary",
                "modularity": 0.42,
                "title": "Melanie · Camping",
                "entities": [("Melanie", 0.9), ("  STRASSE  Café ", 0.3)],
            }
        ]
    )
    assert written == 1
    stmts = _command_strings(client)
    in_topic_call = next(c for c in client.command.await_args_list if "IN_TOPIC" in c.args[1])
    assert "MATCH (e:ENTITY {name_norm: row.nn})" in in_topic_call.args[1]
    assert "MERGE (e)-[r:IN_TOPIC]->(tp)" in in_topic_call.args[1]
    assert "SET r.pagerank = row.pr" in in_topic_call.args[1]
    rows = in_topic_call.kwargs["params"]["rows"]
    by_nn = {r["nn"]: r["pr"] for r in rows}
    # casefold (not lower) + whitespace collapse — same key write_entities_bulk uses.
    assert by_nn == {"melanie": 0.9, "strasse café": 0.3}
    # No chunk-evidence edge: HAS_TOPIC had no reader once the topic retrieval
    # channel went, so the write was pure cost.
    assert not any("HAS_TOPIC" in s for s in stmts)


@pytest.mark.asyncio
async def test_write_topics_skips_in_topic_when_no_entities() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_topics([{"topic_id": "topic:abc:1"}])
    assert not any("IN_TOPIC" in c.args[1] for c in client.command.await_args_list)


# ---------------------------------------------------------------------------
# Relation writes — evidence append + tombstoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_relation_evidence_append_sql_shape() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_relation(
        src_name="Melanie",
        predicate="LIKES",
        dst_name="Camping",
        fact="Melanie likes camping",
        chunk_id="chunk-1",
        confidence=0.8,
        source="gliner",
    )
    stmts = _command_strings(client)
    # One REL type carries every predicate as a property, so no per-predicate
    # CREATE EDGE TYPE DDL runs on the ingest path at all.
    assert not any(s.startswith("CREATE EDGE TYPE") for s in stmts)
    merge = next(s for s in stmts if "MERGE (a)-[r:REL {predicate: $predicate}]->(b)" in s)
    # Endpoints merged on name_norm; evidence appended as a deduped quoted literal.
    assert "MERGE (a:ENTITY {name_norm: $src_nn})" in merge
    assert "WHEN 'chunk-1' IN coalesce(r.evidence, []) THEN r.evidence" in merge
    assert "ELSE coalesce(r.evidence, []) + ['chunk-1']" in merge
    # Golden naming: confidence / valid_at / is_current / state / source
    for fragment in ("r.confidence", "r.valid_at", "r.is_current = true", "r.state", "r.source"):
        assert fragment in merge
    merge_call = next(c for c in client.command.await_args_list if "MERGE (a)-[r:REL" in c.args[1])
    assert merge_call.kwargs["params"]["src_nn"] == "melanie"
    assert merge_call.kwargs["params"]["dst_nn"] == "camping"
    assert merge_call.kwargs["params"]["predicate"] == "LIKES"
    # LIKES is multi-valued: no tombstone pass
    assert not any("r.invalid_at = $now" in s for s in stmts)


@pytest.mark.asyncio
async def test_write_relation_single_valued_tombstones() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_relation(
        src_name="Melanie",
        predicate="LIVES_IN",
        dst_name="Toronto",
        chunk_id="chunk-2",
    )
    stmts = _command_strings(client)
    tombstone = next(s for s in stmts if "r.invalid_at = $now" in s)
    assert "is_current: true" in tombstone
    assert "r.is_current = false" in tombstone
    # Scoped by the predicate PROPERTY, not a per-predicate edge type.
    assert "[r:REL {predicate: $predicate, is_current: true}]" in tombstone
    tomb_call = next(
        c for c in client.command.await_args_list if "r.invalid_at = $now" in c.args[1]
    )
    assert tomb_call.kwargs["params"]["predicate"] == "LIVES_IN"


@pytest.mark.asyncio
async def test_write_relation_accepts_predicate_that_is_not_a_valid_type_name() -> None:
    """A predicate is data, not a type name, so it need not be a legal identifier.

    Previously the predicate became the edge TYPE, so anything unusable as DDL
    (leading digit, punctuation) was silently DROPPED — losing the fact outright.
    """
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_relation(src_name="a", predicate="123!!", dst_name="b")
    merge_call = next(c for c in client.command.await_args_list if "MERGE (a)-[r:REL" in c.args[1])
    assert merge_call.kwargs["params"]["predicate"] == "123!!"


@pytest.mark.asyncio
async def test_write_relation_skips_empty_predicate() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_relation(src_name="a", predicate="   ", dst_name="b")
    client.command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Frame-instance writes — PLAYS_ROLE + FRAME_EVOKED_IN + convenience relations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_frame_instances_edges_and_convenience_relations() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    # A 3-role instance: Employee (agent-like) + Position + Employer. The `id` is
    # content-hashed at write time (any input id is ignored).
    await store.write_frame_instances(
        session_id="s1",
        chunk_id="c1",
        instances=[
            {
                "frame": "Being_employed",
                "match_sim": 0.42,
                "role_confidence": 0.95,
                "roles": {
                    "Employee": ["Caroline"],
                    "Position": ["nurse"],
                    "Employer": ["hospital"],
                },
            }
        ],
    )
    stmts = _command_strings(client)

    # FRAME_INSTANCE vertex + FRAME_EVOKED_IN evidence edge, batched via UNWIND.
    fi_call = next(
        c for c in client.command.await_args_list if "MERGE (fi:FRAME_INSTANCE" in c.args[1]
    )
    assert "UNWIND $rows AS row" in fi_call.args[1]
    assert "MERGE (fi:FRAME_INSTANCE {id: row.fid})" in fi_call.args[1]
    assert "MERGE (fi)-[:FRAME_EVOKED_IN]->(c)" in fi_call.args[1]
    inst_rows = fi_call.kwargs["params"]["rows"]
    assert len(inst_rows) == 1
    # Content-hash id: same (frame + role fills) -> same node across chunks.
    assert inst_rows[0]["fid"].startswith("Being_employed:")

    # PLAYS_ROLE edges batched in one UNWIND — 3 roles, 1 filler each.
    role_call = next(
        c for c in client.command.await_args_list if "PLAYS_ROLE {role: row.role}" in c.args[1]
    )
    role_rows = role_call.kwargs["params"]["rows"]
    assert len(role_rows) == 3
    assert {r["role"] for r in role_rows} == {"Employee", "Position", "Employer"}
    assert all(r["fid"] == inst_rows[0]["fid"] for r in role_rows)

    # The reified PLAYS_ROLE form is the ONLY representation: no flattened
    # ENTITY->ENTITY projection is written. The old behaviour emitted
    # f"{FRAME}_{ROLE}" convenience edges (BEING_EMPLOYED_POSITION, ...), which
    # minted one edge TYPE per frame/role pair and re-stated the PLAYS_ROLE facts
    # in a second format that the fact-sheet dedup could not match.
    assert not [s for s in stmts if "MERGE (a)-[r:" in s]


@pytest.mark.asyncio
async def test_write_frame_instances_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    assert await store.write_frame_instances("s1", "c1", []) == 0
    client.command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Read API — state filter threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_chunks_ann_filters_state_and_session() -> None:
    rows = [
        {
            "@rid": "#1:0",
            "chunk_id": "c1",
            "session_id": "s1",
            "state": "raw",
            "text": "a",
            "distance": 0.1,
        },
        {
            "@rid": "#1:1",
            "chunk_id": "c2",
            "session_id": "s1",
            "state": "consolidated",
            "text": "b",
            "distance": 0.2,
        },
        {
            "@rid": "#1:2",
            "chunk_id": "c3",
            "session_id": "s2",
            "state": "raw",
            "text": "c",
            "distance": 0.3,
        },
    ]
    client = _mock_client(query_rows=rows)
    store = GraphStore(client=client)
    out = await store.search_chunks_ann([0.1] * 8, top_k=10, session_id="s1", state="raw")
    assert [r["chunk_id"] for r in out] == ["c1"]
    assert out[0]["cosine_score"] == pytest.approx(0.9)
    # vectorNeighbors SQL over the CHUNK embedding index
    sql = client.query.await_args_list[0].args[1]
    assert 'vectorNeighbors("CHUNK[embedding]"' in sql


@pytest.mark.asyncio
async def test_full_text_search_threads_state_filter_into_cypher() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.full_text_search_chunks(["camping"], session_id="s1", state="raw")
    cypher = client.query.await_args_list[0].args[1]
    assert "toLower(c.text) CONTAINS 'camping'" in cypher
    assert "c.session_id = 's1'" in cypher
    assert "c.state = 'raw'" in cypher


@pytest.mark.asyncio
async def test_query_entity_chunks_threads_state_and_escapes_quotes() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.query_entity_chunks(["O'Melanie", "war"], state="consolidated")
    cypher = client.query.await_args_list[0].args[1]
    # sanitize() escapes rather than strips: ENTITY.name_norm only casefolds, so
    # the stored key keeps its apostrophe ("o'melanie"). Stripping produced
    # 'omelanie', which could never match what was written.
    assert r"e.name_norm CONTAINS 'o\'melanie'" in cypher
    assert "e.name_norm = 'war'" in cypher
    assert "c.state = 'consolidated'" in cypher


@pytest.mark.asyncio
async def test_query_entity_chunks_neutralises_backslash() -> None:
    """A trailing backslash must not escape the closing quote of the literal."""
    client = _mock_client()
    store = GraphStore(client=client)
    await store.query_entity_chunks(["back\\slash"], state=None)
    cypher = client.query.await_args_list[0].args[1]
    assert r"'back\\slash'" in cypher


@pytest.mark.asyncio
async def test_query_temporal_chunks_year_prefix_and_state() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.query_temporal_chunks(["2023", "2023-05-07"], state="raw")
    cypher = client.query.await_args_list[0].args[1]
    assert "d.date = '2023'" in cypher
    assert "d.date STARTS WITH '2023-'" in cypher
    assert "d.date = '2023-05-07'" in cypher
    assert "c.state = 'raw'" in cypher
    assert "MENTIONS_DATE" in cypher


@pytest.mark.asyncio
async def test_core_ddl_frame_carries_summary_and_vector_index() -> None:
    joined = " | ".join(stmt for _, stmt in _CORE_DDL)
    assert "CREATE PROPERTY FRAME.summary" in joined
    assert "CREATE PROPERTY FRAME.embedding" in joined
    assert "ON FRAME(embedding) LSM_VECTOR" in joined


@pytest.mark.asyncio
async def test_delete_session_leaves_canonical_frames_alone() -> None:
    """A canonical FRAME is namespace-global, like ENTITY/TOPIC/TEMPORAL.

    It carries no ``session_id``, so the old per-session clause matched nothing;
    and it must not be deleted anyway — other sessions evoke the same frame, and
    it may be a parent in the BROADER taxonomy. Its evidence edges go when the
    chunks do.
    """
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.delete_session("s1")
    deletes = [c.args[1] for c in client.command.await_args_list if "DETACH DELETE" in c.args[1]]
    assert not [d for d in deletes if "(n:FRAME " in d or "(n:FRAME{" in d]
    # the session-scoped vertices are still purged
    assert any("FRAME_INSTANCE" in d for d in deletes)
    assert any("(n:CHUNK " in d for d in deletes)


def test_core_ddl_declares_the_broader_taxonomy_edge() -> None:
    joined = " | ".join(stmt for _, stmt in _CORE_DDL)
    assert "CREATE EDGE TYPE BROADER" in joined


def test_core_ddl_frame_drops_the_columns_nothing_ever_wrote() -> None:
    """``lu``/``speaker``/``fes``/``fillers`` were written empty on every row.

    The role-filled frame data that matters is reified on FRAME_INSTANCE and
    PLAYS_ROLE, which has a live reader. ``session_id`` goes with them: a
    canonical FRAME spans every session that evokes it.
    """
    joined = " | ".join(stmt for _, stmt in _CORE_DDL)
    for dead in ("FRAME.lu", "FRAME.speaker", "FRAME.fes", "FRAME.fillers", "FRAME.session_id"):
        assert dead not in joined, f"{dead} has no reader and no writer"


@pytest.mark.asyncio
async def test_write_frames_persists_summary_and_embedding() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frames(
        [
            {
                "frame_type": "Process_start",
                "chunk_id": "c1",
                "summary": "the beginning of a process",
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
    )
    cmds = _command_strings(client)
    # summary rides the UNWIND MERGE as a param
    merge = client.command.await_args_list[0]
    assert "n.summary = r.summary" in merge.args[1]
    assert merge.kwargs["params"]["rows"][0]["summary"] == "the beginning of a process"
    # embedding written as an inline literal via _set_embedding on FRAME(id)
    emb_stmt = next(s for s in cmds if "SET n.embedding = [" in s and "FRAME" in s)
    assert "Process_start" not in emb_stmt  # id is a bound $key param, not interpolated
    emb_call = next(c for c in client.command.await_args_list if "SET n.embedding = [" in c.args[1])
    assert emb_call.kwargs["params"]["key"] == "Process_start"


@pytest.mark.asyncio
async def test_write_frames_merges_one_canonical_vertex_per_frame_type() -> None:
    """FRAME is canonical: one vertex per frame, whatever chunk or session evoked it.

    Per-chunk data lives on the EVOKED_BY edge. Without this the graph holds one
    FRAME per (chunk, frame), which is what made frame-to-frame traversal
    impossible and forced coverage to be measured off the node.
    """
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frames([{"frame_type": "Process_start", "chunk_id": "c1", "confidence": 0.9}])
    await store.write_frames([{"frame_type": "Process_start", "chunk_id": "c2", "confidence": 0.4}])

    merges = [c for c in client.command.await_args_list if "MERGE (n:FRAME" in c.args[1]]
    assert [r["id"] for c in merges for r in c.kwargs["params"]["rows"]] == [
        "Process_start",
        "Process_start",
    ]
    # The per-chunk signal is the edge, and it differs per chunk.
    edges = [c for c in client.command.await_args_list if "EVOKED_BY" in c.args[1]]
    assert [r["conf"] for c in edges for r in c.kwargs["params"]["rows"]] == [0.9, 0.4]
    # Nothing per-chunk or dead is set on the canonical node.
    node_sql = merges[0].args[1]
    for dead in ("n.session_id", "n.lu", "n.speaker", "n.fes", "n.fillers"):
        assert dead not in node_sql


@pytest.mark.asyncio
async def test_write_frames_merges_broader_parents_as_taxonomy() -> None:
    """A frame's FrameNet parents become BROADER edges: the traversable taxonomy.

    Parents are scaffolding — MERGEd as bare vertices with no evidence edge and
    no embedding, so they never enter the ANN pool on their own account. They
    exist to be walked down from (see the narrower expansion in
    query_frames_by_embedding).
    """
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frames(
        [
            {
                "frame_type": "Revenge",
                "chunk_id": "c1",
                "parents": ["Rewards_and_punishments", "Response"],
            }
        ]
    )
    edge = next(c for c in client.command.await_args_list if "BROADER" in c.args[1])
    assert "MERGE (n)-[:BROADER]->(p)" in edge.args[1]
    assert edge.kwargs["params"]["rows"] == [
        {"id": "Revenge", "parent": "Rewards_and_punishments"},
        {"id": "Revenge", "parent": "Response"},
    ]
    # Scaffolding only: a parent gets no evidence edge and no vector.
    parent_sql = " | ".join(
        c.args[1] for c in client.command.await_args_list if "r.parent" in c.args[1]
    )
    assert "EVOKED_BY" not in parent_sql
    assert "SET n.embedding" not in parent_sql


@pytest.mark.asyncio
async def test_write_frames_skips_taxonomy_when_no_parents() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frames([{"frame_type": "Motion", "chunk_id": "c1"}])
    assert not [c for c in client.command.await_args_list if "BROADER" in c.args[1]]


@pytest.mark.asyncio
async def test_write_frames_writes_each_frame_embedding_once() -> None:
    """The definition embedding is a property of the frame, not of the evocation."""
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    row = {"frame_type": "Process_start", "embedding": [0.1, 0.2, 0.3]}
    await store.write_frames([{**row, "chunk_id": "c1"}])
    await store.write_frames([{**row, "chunk_id": "c2"}])
    emb = [c for c in client.command.await_args_list if "SET n.embedding = [" in c.args[1]]
    assert len(emb) == 1


@pytest.mark.asyncio
async def test_emptying_the_database_reinstates_the_frame_writes() -> None:
    """The write-once cache must not outlive the rows it is tracking.

    ``clear_all`` deletes every FRAME. A cache that still claims they are
    written makes the next ingest skip their embedding and their taxonomy, and
    an embedding-less FRAME is invisible to the channel's vector search — the
    layer goes quietly dead rather than failing.
    """
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    row = {"frame_type": "Process_start", "chunk_id": "c1", "embedding": [0.1, 0.2]}
    await store.write_frames([row])
    await store.clear_all()
    await store.write_frames([row])
    emb = [c for c in client.command.await_args_list if "SET n.embedding = [" in c.args[1]]
    assert len(emb) == 2


@pytest.mark.asyncio
async def test_ensure_schema_reinstates_the_frame_writes() -> None:
    """Same hazard by the other route: the namespace was dropped and recreated."""
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    row = {"frame_type": "Process_start", "chunk_id": "c1", "embedding": [0.1, 0.2]}
    await store.write_frames([row])
    await store.ensure_schema(dims=2)
    await store.write_frames([row])
    emb = [c for c in client.command.await_args_list if "SET n.embedding = [" in c.args[1]]
    assert len(emb) == 2


@pytest.mark.asyncio
async def test_query_frames_by_embedding_ann_then_evoked_by_join() -> None:
    """Session scoping is on the CHUNK, and the evocation strength weights the score.

    FRAME is canonical, so it carries no session — filtering it there would drop
    every frame that happened to be first evoked in another session. The chunk
    is what belongs to a session.
    """
    client = _mock_client()
    # ANN pass, then the narrower-expansion pass, then the EVOKED_BY join.
    client.query = AsyncMock(
        side_effect=[
            [{"id": "Process_start", "frame_type": "Process_start", "distance": 0.6}],
            [],
            [
                {
                    "id": "Process_start",
                    "frame_type": "Process_start",
                    "chunk_id": "c1",
                    "confidence": 0.5,
                },
                {
                    "id": "Process_start",
                    "frame_type": "Process_start",
                    "chunk_id": "c2",
                    "confidence": 1.0,
                },
            ],
        ]
    )
    store = GraphStore(client=client, db="mem")
    rows = await store.query_frames_by_embedding([0.1, 0.2], top_k=5, session_id="s1")
    ann_sql = client.query.await_args_list[0].args[1]
    assert 'vectorNeighbors("FRAME[embedding]"' in ann_sql
    join_cypher = client.query.await_args_list[-1].args[1]
    assert "(n:FRAME)-[ev:EVOKED_BY]->(c:CHUNK)" in join_cypher
    assert "c.session_id = 's1'" in join_cypher
    assert "n.session_id" not in join_cypher
    # 1 - 0.6 = 0.4 query match, scaled by how strongly each chunk evoked it.
    assert rows == [
        {"chunk_id": "c1", "frame_type": "Process_start", "score": pytest.approx(0.2)},
        {"chunk_id": "c2", "frame_type": "Process_start", "score": pytest.approx(0.4)},
    ]


@pytest.mark.asyncio
async def test_query_frames_by_embedding_expands_to_narrower_frames() -> None:
    """A query matching a parent frame reaches chunks that only evoked its child.

    This is the reader that makes the BROADER taxonomy worth writing. Without
    it the channel only ever returns chunks whose frame the query matched
    head-on, and the frame hierarchy is a write-only layer.
    """
    client = _mock_client()
    client.query = AsyncMock(
        side_effect=[
            [
                {
                    "id": "Rewards_and_punishments",
                    "frame_type": "Rewards_and_punishments",
                    "distance": 0.2,
                }
            ],
            [{"child": "Revenge", "parent": "Rewards_and_punishments"}],
            [{"id": "Revenge", "frame_type": "Revenge", "chunk_id": "c9", "confidence": 1.0}],
        ]
    )
    store = GraphStore(client=client, db="mem")
    rows = await store.query_frames_by_embedding([0.1, 0.2], top_k=5)
    narrower = client.query.await_args_list[1].args[1]
    assert "(child:FRAME)-[:BROADER]->(parent:FRAME)" in narrower
    assert "Rewards_and_punishments" in narrower
    # The child joins the pool, so its chunk is reachable...
    assert "Revenge" in client.query.await_args_list[2].args[1]
    # ...at a discount on the parent's 0.8 match, so a direct hit always outranks it.
    assert rows == [
        {"chunk_id": "c9", "frame_type": "Revenge", "score": pytest.approx(0.8 * 0.6)},
    ]


# ---------------------------------------------------------------------------
# Ontology taxonomy (SKOS broader)
# ---------------------------------------------------------------------------


class _FakeTerm:
    kind = "class"
    definition = "a definition"
    module = "AgentOntology"

    def __init__(self, uri: str, label: str, parents: tuple[str, ...] = ()) -> None:
        self.uri = uri
        self.label = label
        self.parents = list(parents)


class _FakeIndex:
    digest = "digest-1"
    source = "cco"

    def __init__(self, terms: list[_FakeTerm]) -> None:
        self.terms = terms

    def term_embedding(self, uri: str) -> list[float]:
        return []


@pytest.mark.asyncio
async def test_write_ontology_persists_broader_edges_keyed_by_uri() -> None:
    """The class hierarchy lands as edges, not only as a comma-joined string.

    Parents are DECLARED as labels but resolved to URIs before the MERGE: the
    label index is NOTUNIQUE, so matching the parent by label would fan the edge
    out across every class sharing that label.
    """
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_ontology(
        _FakeIndex(
            [
                _FakeTerm("u:studio", "Dance Studio", ("Facility",)),
                _FakeTerm("u:facility", "Facility"),
            ]
        )
    )
    edge = next(c for c in client.command.await_args_list if "BROADER" in c.args[1])
    assert "MERGE (c)-[:BROADER]->(p)" in edge.args[1]
    assert "{uri: r.puri}" in edge.args[1]
    assert edge.kwargs["params"]["rows"] == [{"uri": "u:studio", "puri": "u:facility"}]


@pytest.mark.asyncio
async def test_write_ontology_backfills_broader_on_an_existing_namespace() -> None:
    """A namespace that already has the terms still needs the edges.

    ``write_ontology`` short-circuits on a term count guard, and the subclass
    expansion now reads ONLY BROADER. Without a backfill, every namespace
    ingested before this feature would keep its classes and silently stop
    generalising — no error, no log, just an empty expansion.
    """
    client = _mock_client()
    # Term guard is satisfied (2 terms already present) but no BROADER edges yet.
    client.query = AsyncMock(side_effect=[[{"c": 2}], [{"c": 0}]])
    store = GraphStore(client=client, db="mem")
    await store.write_ontology(
        _FakeIndex(
            [
                _FakeTerm("u:studio", "Dance Studio", ("Facility",)),
                _FakeTerm("u:facility", "Facility"),
            ]
        )
    )
    edge = next(c for c in client.command.await_args_list if "BROADER" in c.args[1])
    assert edge.kwargs["params"]["rows"] == [{"uri": "u:studio", "puri": "u:facility"}]
    # ...but it must NOT re-MERGE the 1401 class rows to do it.
    assert not [c for c in client.command.await_args_list if "MERGE (n:ONTOLOGY_CLASS" in c.args[1]]


@pytest.mark.asyncio
async def test_search_ontology_classes_expands_to_subclasses_via_broader() -> None:
    """A query matching ``Facility`` reaches a chunk tagged only ``Dance Studio``.

    The expansion used to full-scan every ONTOLOGY_CLASS row and re-split the
    ``parents`` string in Python. Traversing BROADER is one indexed hop, matches
    on URI rather than on a label that two classes can share, and gives the
    persisted hierarchy a live reader.
    """
    client = _mock_client()
    client.query = AsyncMock(
        side_effect=[
            [{"uri": "u:facility", "label": "Facility", "distance": 0.1}],
            [{"child": "u:studio", "label": "Dance Studio", "parent": "u:facility"}],
            [{"chunk_id": "c1", "uri": "u:studio", "sim": 1.0}],
        ]
    )
    store = GraphStore(client=client, db="mem")
    rows = await store.search_ontology_classes([0.1], top_k=5)
    broader_q = client.query.await_args_list[1].args[1]
    assert "(child:ONTOLOGY_CLASS)-[:BROADER]->(parent:ONTOLOGY_CLASS)" in broader_q
    assert "u:facility" in broader_q
    assert rows == [
        {
            "chunk_id": "c1",
            "uri": "u:studio",
            "label": "Dance Studio",
            "score": pytest.approx(0.9 * 0.6),
            "via": "chunk",
        }
    ]


# ---------------------------------------------------------------------------
# Structured fact context (D8 fact-sheet)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_fact_context_formats_filters_prioritises_and_caps() -> None:
    # The rel query returns r.predicate — the RAW label. Predicates are data, so a
    # reserved-word verb ("find") needs no escaping and reads back verbatim.
    # Frame facts arrive ONLY via frame_rows (the reified PLAYS_ROLE layer); the
    # flattened f"{FRAME}_{ROLE}" projection is no longer written, so no predicate
    # here restates a frame — which is what the old noise filter existed to undo.
    rel_rows = [
        {"s": "Gina", "p": "SUPPORT", "o": "Jon", "ev": ["c2"]},
        {
            "s": "Gina",
            "p": "FIND",
            "o": "fashion pieces",
            "ev": ["c3"],
        },  # reserved-word verb → KEPT
        {"s": "Jon", "p": "VISITING", "o": "Rome", "ev": []},  # plain (unevidenced)
    ]
    frame_rows = [
        {"frame": "Activity_start", "roles": ["Agent=Jon", "Activity=dance studio"]},
        {"frame": "Visiting", "roles": ["Agent=Jon"]},  # < 2 role fills → dropped
    ]
    client = _mock_client()
    client.query = AsyncMock(side_effect=[rel_rows, frame_rows])
    store = GraphStore(client=client)

    facts = await store.entity_fact_context(["Jon", "Gina"])

    # Frame instance (>=2 roles) ranks first; the 1-role frame is dropped.
    assert facts[0] == "Activity_start: Agent=Jon, Activity=dance studio"
    assert not any(f.startswith("Visiting:") for f in facts)
    assert "Gina support Jon" in facts
    assert "Gina find fashion pieces" in facts  # reserved-word verb survived
    assert "Jon visiting Rome" in facts
    # Evidenced relations precede the bare (unevidenced) VISITING relation.
    assert facts.index("Gina support Jon") < facts.index("Jon visiting Rome")

    # Query: both directions, current-only, raw predicate, name_norm list literal.
    rel_cypher = client.query.await_args_list[0].args[1]
    assert "a.name_norm IN ['gina','jon'] OR b.name_norm IN ['gina','jon']" in rel_cypher
    assert "r.is_current = true OR r.is_current IS NULL" in rel_cypher
    assert "r.predicate AS p" in rel_cypher
    assert "(a:ENTITY)-[r:REL]->(b:ENTITY)" in rel_cypher
    # Structural edges are separate types, so scoping to REL excludes them by
    # construction — the old exclusion list is gone.
    assert "NOT type(r) IN [" not in rel_cypher


@pytest.mark.asyncio
async def test_entity_fact_context_caps_at_max_facts() -> None:
    rel_rows = [{"s": "Jon", "p": "SUPPORT", "o": f"person{i}", "ev": ["c"]} for i in range(10)]
    client = _mock_client()
    client.query = AsyncMock(side_effect=[rel_rows, []])
    store = GraphStore(client=client)
    facts = await store.entity_fact_context(["Jon"], max_facts=3)
    assert len(facts) == 3


@pytest.mark.asyncio
async def test_entity_fact_context_empty_on_no_entities() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    assert await store.entity_fact_context([]) == []
    client.query.assert_not_called()


@pytest.mark.asyncio
async def test_entity_fact_context_soft_fails_on_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=RuntimeError("boom"))
    store = GraphStore(client=client)
    assert await store.entity_fact_context(["Jon"]) == []


@pytest.mark.asyncio
async def test_resolve_query_entities_matches_word_boundary() -> None:
    client = _mock_client(query_rows=[{"name": "Jon", "nn": "jon"}, {"name": "Gina", "nn": "gina"}])
    store = GraphStore(client=client)
    names = await store.resolve_query_entities(["jon", "studio", "of"])  # "of" too short
    assert names == ["Jon", "Gina"]
    cypher = client.query.await_args_list[0].args[1]
    # Word-boundary: exact, or a whole space-delimited word — NOT intra-word substring
    # (so "there" never resolves "theresa").
    assert "e.name_norm = 'jon'" in cypher
    assert "e.name_norm ENDS WITH ' studio'" in cypher  # "studio" -> "dance studio"
    assert "e.name_norm STARTS WITH 'studio '" in cypher
    assert "e.name_norm CONTAINS ' studio '" in cypher
    assert "CONTAINS 'studio'" not in cypher  # never a bare intra-word substring
    assert "'of'" not in cypher  # < 3 chars filtered out
    # The cap is applied AFTER ranking, so the query must not pre-truncate.
    assert "LIMIT" not in cypher


@pytest.mark.asyncio
async def test_resolve_query_entities_ranks_before_the_cap() -> None:
    """The cap binds on every real query, so WHICH entities survive it is the

    whole behaviour. Storage order used to decide; exact-then-shortest decides
    now. Rows are given worst-first so a pass-through would fail.
    """
    client = _mock_client(
        query_rows=[
            {
                "name": "Gina's hard work and effort, Gina",
                "nn": "gina's hard work and effort, gina",
            },
            {"name": "dance studio", "nn": "dance studio"},
            {"name": "Gina, Gina", "nn": "gina, gina"},
            {"name": "Gina", "nn": "gina"},
            {"name": "studio", "nn": "studio"},
        ]
    )
    store = GraphStore(client=client)
    names = await store.resolve_query_entities(["gina", "studio"], cap=3)
    # Exact token matches lead; the rest fall back to shortest-first.
    assert names == ["Gina", "studio", "Gina, Gina"]
    # What actually matters: the canonical name outranks its corrupted variants,
    # which is what the unordered LIMIT could not guarantee.
    assert names.index("Gina") < names.index("Gina, Gina")
    assert "Gina's hard work and effort, Gina" not in names


# ---------------------------------------------------------------------------
# Bulk chunk writes — batching cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_chunks_bulk_batches_at_cap() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    n = _CHUNK_BATCH * 2 + 5  # 45 with the default cap of 20 → 3 batches
    chunks = [
        {"chunk_id": f"c{i}", "session_id": "s1", "file_id": "f1", "text": f"t{i}", "seq": i}
        for i in range(n)
    ]
    await store.write_chunks_bulk(chunks)
    merge_calls = [
        c
        for c in client.command.await_args_list
        if "MERGE (c:CHUNK {chunk_id: row.cid})" in c.args[1]
    ]
    assert len(merge_calls) == 3
    assert [len(c.kwargs["params"]["rows"]) for c in merge_calls] == [
        _CHUNK_BATCH,
        _CHUNK_BATCH,
        5,
    ]
    # Each batch also writes PART_OF edges; state lands 'raw' on create.
    edge_calls = [c for c in client.command.await_args_list if "PART_OF" in c.args[1]]
    assert len(edge_calls) == 3
    assert "coalesce(c.state, 'raw')" in merge_calls[0].args[1]


@pytest.mark.asyncio
async def test_write_chunks_bulk_links_each_chunk_to_its_session() -> None:
    """CHUNK-[:IN_SESSION]->SESSION, from the session_id already on every row.

    TURN has carried this edge since the buffer landed; CHUNK never did, so a
    chunk reached its session only through CHUNK-[:PART_OF]->FILE-[:IN_SESSION]
    ->SESSION — which does not exist at all for a chunk written with no
    file_id. Every traversal that wanted 'the session this memory came from'
    had to fall back to matching on the string property instead.
    """
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_chunks_bulk(
        [
            {"chunk_id": "c1", "session_id": "s1", "text": "t1"},  # no file_id
            {"chunk_id": "c2", "session_id": "", "text": "t2"},  # no session either
        ]
    )

    session_edges = [c for c in client.command.await_args_list if "IN_SESSION" in c.args[1]]
    assert len(session_edges) == 1
    # Only the chunk that HAS a session is linked; a sessionless chunk must not
    # MERGE a SESSION vertex with an empty id into existence.
    assert [r["cid"] for r in session_edges[0].kwargs["params"]["rows"]] == ["c1"]
    assert "MATCH (s:SESSION {session_id: row.sid})" in session_edges[0].args[1]


@pytest.mark.asyncio
async def test_write_chunk_embedding_is_inline_literal() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_chunk(
        {
            "chunk_id": "c1",
            "session_id": "s1",
            "file_id": "f1",
            "text": "t",
            "embedding": [0.5, -0.25],
        }
    )
    emb_stmt = next(s for s in _command_strings(client) if "n.embedding =" in s)
    # Inline SQL literal, never a parameter (LSM_VECTOR constraint).
    assert "n.embedding = [0.5,-0.25]" in emb_stmt


# ---------------------------------------------------------------------------
# Entity resolution primitives (consolidation / flush path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_entities_counts_mentions() -> None:
    client = _mock_client(
        query_rows=[
            {"name_norm": "melanie", "name": "Melanie", "type": "PERSON", "mention_count": 3},
            {"name_norm": "others", "name": "others", "type": None, "mention_count": 0},
        ]
    )
    store = GraphStore(client=client)
    out = await store.list_entities()
    assert out == [
        {"name_norm": "melanie", "name": "Melanie", "type": "PERSON", "mention_count": 3},
        {"name_norm": "others", "name": "others", "type": "", "mention_count": 0},
    ]
    cypher = client.query.await_args_list[0].args[1]
    assert "OPTIONAL MATCH (c:CHUNK)-[:MENTIONS]->(e)" in cypher
    assert "count(c) AS mention_count" in cypher


@pytest.mark.asyncio
async def test_drop_entities_detach_deletes_by_name_norm() -> None:
    client = _mock_client(query_rows=[{"n": 2}])
    store = GraphStore(client=client)
    n = await store.drop_entities(["others", "kind"])
    assert n == 2
    delete_stmt = next(s for s in _command_strings(client) if "DETACH DELETE e" in s)
    assert "e.name_norm IN ['others','kind']" in delete_stmt


@pytest.mark.asyncio
async def test_drop_entities_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    assert await store.drop_entities([]) == 0
    client.command.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_entities_repoints_mentions_relations_and_deletes_loser() -> None:
    client = _mock_client()
    client.query = AsyncMock(
        side_effect=[
            [{"name": "Mel", "aliases": []}],  # loser lookup
            [{"name": "Melanie", "aliases": []}],  # winner lookup
            [  # outgoing loser relation edges
                {
                    "dst": "Camping",
                    "dst_nn": "camping",
                    "predicate": "LIKES",
                    "fact": "",
                    "evidence": ["c1"],
                    "confidence": 0.8,
                    "valid_at": None,
                    "source": "gliner",
                },
            ],
            [],  # incoming loser relation edges
        ]
    )
    store = GraphStore(client=client)
    merged = await store.merge_entities("melanie", "mel")
    assert merged is True

    stmts = _command_strings(client)
    # MENTIONS re-pointed onto the winner.
    assert any("MERGE (c)-[nm:MENTIONS]->(w)" in s for s in stmts)
    # Relation re-created on the winner via write_relation (REL/LIKES edge).
    merge_rel = next(s for s in stmts if "MERGE (a)-[r:REL {predicate: $predicate}]->(b)" in s)
    rel_call = next(c for c in client.command.await_args_list if "MERGE (a)-[r:REL" in c.args[1])
    assert rel_call.kwargs["params"]["src_nn"] == "melanie"
    assert rel_call.kwargs["params"]["dst_nn"] == "camping"
    assert rel_call.kwargs["params"]["predicate"] == "LIKES"
    assert "chunk_id" not in merge_rel or "c1" in str(rel_call.kwargs["params"])
    # Loser surface folded into winner aliases.
    alias_call = next(
        c for c in client.command.await_args_list if "SET w.aliases = $aliases" in c.args[1]
    )
    assert "Mel" in alias_call.kwargs["params"]["aliases"]
    # Loser removed.
    assert any("DETACH DELETE lo" in s for s in stmts)


@pytest.mark.asyncio
async def test_merge_entities_noop_when_loser_missing() -> None:
    client = _mock_client()
    client.query = AsyncMock(return_value=[])  # loser lookup returns nothing
    store = GraphStore(client=client)
    assert await store.merge_entities("melanie", "mel") is False
    client.command.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_entities_noop_on_same_node() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    assert await store.merge_entities("melanie", "melanie") is False
    client.query.assert_not_awaited()
    client.command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Graph analytics (WS4) — native ArcadeDB algos + numpy pagerank fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_analytical_view_creates_and_rebuilds() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store._ensure_analytical_view()
    stmts = _command_strings(client)
    assert any("CREATE GRAPH ANALYTICAL VIEW mem_olap UPDATE MODE OFF" in s for s in stmts)
    assert any("REBUILD GRAPH ANALYTICAL VIEW mem_olap" in s for s in stmts)
    # Both statements ride the SQL endpoint (CALL is cypher-only, DDL is sql-only).
    assert all(c.kwargs.get("language") == "sql" for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_ensure_analytical_view_swallows_already_exists() -> None:
    client = _mock_client()

    async def command_side_effect(db, cmd, language="cypher", params=None, session_id=None):
        if cmd.startswith("CREATE GRAPH ANALYTICAL VIEW"):
            raise Exception("view already exists")
        return []

    client.command = AsyncMock(side_effect=command_side_effect)
    store = GraphStore(client=client)
    await store._ensure_analytical_view()  # must not raise
    assert any("REBUILD GRAPH ANALYTICAL VIEW" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_ensure_analytical_view_raises_on_real_error() -> None:
    client = _mock_client()
    client.command = AsyncMock(side_effect=Exception("connection refused"))
    store = GraphStore(client=client)
    with pytest.raises(Exception, match="connection refused"):
        await store._ensure_analytical_view()


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_native_success() -> None:
    client = _mock_client(
        query_rows=[{"nn": "melanie", "rank": 0.42}, {"nn": "caroline", "rank": 0.15}]
    )
    store = GraphStore(client=client)
    updated = await store.compute_and_persist_pagerank()
    assert updated == 2
    # ArcadeDB yields the pagerank column as `score`; YIELDing `rank` binds an
    # unknown field that comes back null and silently forces the numpy fallback.
    assert any(
        "CALL algo.pageRank()" in c.args[1] and "YIELD node, score" in c.args[1]
        for c in client.query.await_args_list
    )
    # Persisted in one batched UNWIND, not a per-entity SET loop.
    set_calls = [c for c in client.command.await_args_list if "SET e.pagerank = r.v" in c.args[1]]
    assert len(set_calls) == 1
    prs = sorted(r["v"] for r in set_calls[0].kwargs["params"]["rows"])
    assert prs == [0.15, 0.42]


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_soft_fails_on_native_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(
        side_effect=Exception("CommandSQLParsingException: algo.pageRank unsupported")
    )
    store = GraphStore(client=client)
    assert await store.compute_and_persist_pagerank() == 0
    assert not any("SET e.pagerank = r.v" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_soft_fails_when_all_ranks_null() -> None:
    """Regression guard: on a real ~190-entity ArcadeDB 26.5.1 graph,
    ``CALL algo.pageRank()`` returns rows with every ``rank`` null (verified
    live during WS4) — that must be treated as a failure, not persisted as
    meaningless zeros.
    """
    client = _mock_client(
        query_rows=[{"nn": "melanie", "rank": None}, {"nn": "caroline", "rank": None}]
    )
    store = GraphStore(client=client)
    assert await store.compute_and_persist_pagerank() == 0
    assert not any("SET e.pagerank = r.v" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_skip_view_ensure_avoids_rebuild() -> None:
    client = _mock_client(query_rows=[{"nn": "melanie", "rank": 0.5}])
    store = GraphStore(client=client)
    await store.compute_and_persist_pagerank(skip_view_ensure=True)
    assert not any("GRAPH ANALYTICAL VIEW" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_graph_embeddings_writes_list_property() -> None:
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "algo.node2vec" in cypher:
            return [{"nn": "melanie", "embedding": [0.1, 0.2, 0.3]}]
        if "(a:ENTITY)-[r:REL]->(b:ENTITY)" in cypher:
            return [{"nn": "melanie", "deg": 2}]
        if "(a:ENTITY)<-[r:REL]-(b:ENTITY)" in cypher:
            return [{"nn": "melanie", "deg": 2}]
        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    store = GraphStore(client=client)
    updated = await store.compute_graph_embeddings()
    assert updated == 1
    # Inline literal, not a $param (ArcadeDB double-wraps LIST-typed params in
    # SET — verified live; see the _emb_sql usage in the source).
    set_call = next(
        c for c in client.command.await_args_list if "e.graph_embedding = [" in c.args[1]
    )
    assert "e.graph_embedding = [0.1,0.2,0.3]" in set_call.args[1]
    assert set_call.kwargs["params"] == {"nn": "melanie"}


@pytest.mark.asyncio
async def test_compute_graph_embeddings_skips_sparse_entities() -> None:
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "algo.node2vec" in cypher:
            return [{"nn": "loner", "embedding": [0.1, 0.2]}]
        # Degree queries return nothing for "loner" -> total degree 0 < 3.
        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    store = GraphStore(client=client)
    updated = await store.compute_graph_embeddings()
    assert updated == 0
    # View-ensure DDL runs, but no ENTITY was actually written.
    assert not any("e.graph_embedding = [" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_graph_embeddings_soft_fails_on_native_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=Exception("algo.node2vec not supported"))
    store = GraphStore(client=client)
    updated = await store.compute_graph_embeddings()
    assert updated == 0
    assert not any("e.graph_embedding = [" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_communities_writes_community_id() -> None:
    client = _mock_client(query_rows=[{"nn": "melanie", "cid": 2}, {"nn": "caroline", "cid": 2}])
    store = GraphStore(client=client)
    updated = await store.compute_communities()
    assert updated == 2
    set_calls = [
        c for c in client.command.await_args_list if "SET e.community_id = r.v" in c.args[1]
    ]
    assert len(set_calls) == 1  # one batched UNWIND
    assert all(r["v"] == 2 for r in set_calls[0].kwargs["params"]["rows"])


@pytest.mark.asyncio
async def test_compute_communities_soft_fails_to_zero_on_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=Exception("algo.louvain not supported"))
    store = GraphStore(client=client)
    updated = await store.compute_communities()
    assert updated == 0
    assert not any("e.community_id = $cid" in c.args[1] for c in client.command.await_args_list)


@pytest.mark.asyncio
async def test_compute_communities_soft_fails_on_empty_result() -> None:
    client = _mock_client(query_rows=[])
    store = GraphStore(client=client)
    assert await store.compute_communities() == 0
    assert not any("e.community_id = $cid" in c.args[1] for c in client.command.await_args_list)


# ---------------------------------------------------------------------------
# Integration (live ArcadeDB only)
# ---------------------------------------------------------------------------


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/ltm",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_store_round_trip_on_scratch_database() -> None:
    """Full write→read round trip on a scratch mem_test_<uuid> database."""
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client

    s = GraphKnowsSettings()
    client = build_arcadedb_client(s)
    db = f"mem_test_{uuid.uuid4().hex[:8]}"
    store = GraphStore(client=client, db=db)
    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        await store.ensure_schema(dims=8)  # idempotent second run

        await store.ensure_session("sess-1", started_at="2026-07-13T00:00:00Z")
        await store.ensure_file("file-1", "sess-1")

        emb_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        await store.write_chunks_bulk(
            [
                {
                    "chunk_id": "chunk-a",
                    "session_id": "sess-1",
                    "file_id": "file-1",
                    "text": "Melanie went camping in Banff.",
                    "seq": 0,
                    "embedding": emb_a,
                },
                {
                    "chunk_id": "chunk-b",
                    "session_id": "sess-1",
                    "file_id": "file-1",
                    "text": "Caroline moved to Toronto.",
                    "seq": 1,
                    "embedding": emb_b,
                },
            ]
        )
        await store.write_entities_bulk(
            "sess-1",
            "chunk-a",
            [
                {"name": "Melanie", "type": "PERSON", "confidence": 0.9},
                {"name": "Banff", "type": "PLACE", "confidence": 0.8},
            ],
        )
        # Same relation from two chunks → evidence list must accumulate both.
        await store.write_relation(
            "Melanie",
            "LIKES",
            "Camping",
            fact="Melanie likes camping",
            chunk_id="chunk-a",
            confidence=0.7,
            source="gliner",
        )
        await store.write_relation(
            "Melanie",
            "LIKES",
            "Camping",
            chunk_id="chunk-b",
            confidence=0.9,
        )

        # ANN read-back
        ann = await store.search_chunks_ann(emb_a, top_k=2, session_id="sess-1", state="raw")
        assert ann and ann[0]["chunk_id"] == "chunk-a"

        # Entity traversal
        ent = await store.query_entity_chunks(["Melanie"], session_id="sess-1", top_k=5)
        assert any(r["chunk_id"] == "chunk-a" for r in ent)

        # Evidence provenance accumulated across both writes; confidence keeps max.
        # One REL type for every predicate; LIKES is the `predicate` property.
        rows = await store.query(
            "MATCH (a:ENTITY {name_norm: 'melanie'})-[r:REL {predicate: 'LIKES'}]->(b) "
            "RETURN r.evidence AS evidence, r.confidence AS confidence, r.fact AS fact"
        )
        assert rows, "LIKES edge not found"
        assert set(rows[0]["evidence"]) == {"chunk-a", "chunk-b"}
        assert rows[0]["confidence"] == pytest.approx(0.9)
        assert rows[0]["fact"] == "Melanie likes camping"

        # get_chunks_by_ids
        by_id = await store.get_chunks_by_ids(["chunk-a", "chunk-b"])
        assert set(by_id) == {"chunk-a", "chunk-b"}
        assert by_id["chunk-a"]["state"] == "raw"
    finally:
        try:
            await store.drop_database()
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_write_entities_bulk_collapses_duplicate_merge_keys() -> None:
    """Two rows sharing a name_norm must never reach the UNWIND.

    Regression: the UNWIND MERGEd both before the ENTITY[name_norm] UNIQUE index
    settled, ArcadeDB raised DuplicatedKeyException, and the caller's whole
    DOCUMENT failed — silently deleting every chunk in it from the graph. An
    extractor running with flat_ner=False returns overlapping spans, so the same
    surface arrives under several labels routinely.
    """
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_entities_bulk(
        "s1",
        "c1",
        [
            {"name": "Becoming Nicole", "type": "WORK_OF_ART", "score": 0.7},
            {"name": "becoming  nicole", "type": "PERSON", "score": 0.9},  # same merge key
            {"name": "Caroline", "type": "PERSON", "score": 0.8},
        ],
    )

    call = next(c for c in client.command.await_args_list if "MERGE (e:ENTITY" in c.args[1])
    rows = call.kwargs["params"]["rows"]
    keys = [r["nn"] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate merge keys reached the UNWIND"
    assert sorted(keys) == ["becoming nicole", "caroline"]
    # Highest confidence wins the type.
    nicole = next(r for r in rows if r["nn"] == "becoming nicole")
    assert nicole["conf"] == pytest.approx(0.9)
    assert nicole["etype"] == "PERSON"


@pytest.mark.asyncio
async def test_write_frame_instances_never_creates_entity_nodes() -> None:
    """The frame layer attaches to entities; it must never mint them.

    This asserted the OPPOSITE until measured. It used to MERGE each filler as
    an ENTITY before the edge pass, guarding a DuplicatedKeyException when one
    entity filled two roles (Being_named with Entity=X and Name=X). That guard
    is now structurally unnecessary — there is no entity write to duplicate —
    and the MERGE it guarded was itself the defect: on LoCoMo conv-30 it invented
    624 of 2105 ENTITY nodes (30%) with no type, no confidence and no MENTIONS
    edge to any chunk, purely because a frame pointed at a span. Those nodes are
    unreachable by any entity query and cannot be resolved or merged with
    anything, yet they reached the generator's fact-sheet as role fillers
    ("Buyer=Hey Gina!", "Speaker=Yeah").

    A span worth a node is one the EXTRACTION path recognised on its own
    evidence. A role citing a span nothing else saw has no provenance, so the
    edge is dropped rather than the node invented.
    """
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {
                "frame": "Being_named",
                "match_sim": 0.9,
                "role_confidence": 0.9,
                # SAME filler in two roles — the shape that broke ingest.
                "roles": {"Entity": ["Becoming Nicole"], "Name": ["Becoming Nicole"]},
            }
        ],
    )

    stmts = _command_strings(client)

    # No statement anywhere in this path may create an ENTITY.
    assert not [s for s in stmts if "MERGE (e:ENTITY" in s], (
        "the frame path must not mint ENTITY nodes — a filler with no extraction "
        "evidence has no provenance and becomes an unreachable, untyped node"
    )

    # The role edge MATCHes an already-existing entity; a filler the extractor
    # never saw simply matches nothing and the edge is not written.
    edge = next(s for s in stmts if "PLAYS_ROLE {role: row.role}" in s)
    assert "MATCH (e:ENTITY {name_norm: row.nn})" in edge
