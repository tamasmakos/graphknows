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
from graphknows.storage.namespace import db_name, namespaces_from_databases


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


def test_namespaces_from_databases_recognizes_mem_prefix() -> None:
    namespaces = namespaces_from_databases(["mem", "mem_acme", "stm_beta", "ltm_beta", "other"])
    assert namespaces[0] == ""
    assert "acme" in namespaces
    assert "beta" in namespaces
    assert "other" not in namespaces


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------


def test_core_ddl_contains_golden_types_and_not_dead_ones() -> None:
    stmts = [stmt for _, stmt in _CORE_DDL]
    type_stmts = [s for s in stmts if s.startswith(("CREATE VERTEX TYPE", "CREATE EDGE TYPE"))]

    def created(kind: str, name: str) -> bool:
        return any(s.startswith(f"CREATE {kind} TYPE {name} ") for s in type_stmts)

    # Golden vertices
    for vertex in ("SESSION", "FILE", "CHUNK", "ENTITY", "TEMPORAL", "TOPIC", "FRAME"):
        assert created("VERTEX", vertex), f"missing vertex {vertex}"
    # Golden edges
    for edge in ("PART_OF", "IN_SESSION", "MENTIONS", "MENTIONS_DATE", "HAS_TOPIC", "EVOKED_BY"):
        assert created("EDGE", edge), f"missing edge {edge}"
    # Dead schema must NOT be recreated
    for vertex in ("DIALOG", "TURN", "SEGMENT", "DOC", "SESSIONENTITY"):
        assert not created("VERTEX", vertex), f"dead vertex {vertex} present"
    for edge in (
        "HAS_CHUNK",
        "HAS_TURN",
        "HAS_ENTITY",
        "NEXT",
        "IN_TOPIC",
        "CO_OCCURS",
        "APPEARED_IN",
        "CONTAINS_DOC",
        "IN_COMMUNITY",
        "HAS_FILE",
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
async def test_write_entities_bulk_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_entities_bulk(session_id="s1", chunk_id="c1", entities=[])
    client.command.assert_not_awaited()


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
    # Edge type declared before MERGE
    assert any(s.startswith("CREATE EDGE TYPE LIKES") for s in stmts)
    merge = next(s for s in stmts if "MERGE (a)-[r:LIKES]->(b)" in s)
    # Endpoints merged on name_norm; evidence appended as a deduped quoted literal.
    assert "MERGE (a:ENTITY {name_norm: $src_nn})" in merge
    assert "WHEN 'chunk-1' IN coalesce(r.evidence, []) THEN r.evidence" in merge
    assert "ELSE coalesce(r.evidence, []) + ['chunk-1']" in merge
    # Golden naming: confidence / valid_at / is_current / state / source
    for fragment in ("r.confidence", "r.valid_at", "r.is_current = true", "r.state", "r.source"):
        assert fragment in merge
    merge_call = next(
        c for c in client.command.await_args_list if "MERGE (a)-[r:LIKES]->(b)" in c.args[1]
    )
    assert merge_call.kwargs["params"]["src_nn"] == "melanie"
    assert merge_call.kwargs["params"]["dst_nn"] == "camping"
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
    assert "[r:LIVES_IN" in tombstone


@pytest.mark.asyncio
async def test_write_relation_skips_unlabelable_predicate() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_relation(src_name="a", predicate="123!!", dst_name="b")
    client.command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Read API — state filter threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_chunks_ann_filters_state_and_session() -> None:
    rows = [
        {"@rid": "#1:0", "chunk_id": "c1", "session_id": "s1", "state": "raw",
         "text": "a", "distance": 0.1},
        {"@rid": "#1:1", "chunk_id": "c2", "session_id": "s1", "state": "consolidated",
         "text": "b", "distance": 0.2},
        {"@rid": "#1:2", "chunk_id": "c3", "session_id": "s2", "state": "raw",
         "text": "c", "distance": 0.3},
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
async def test_query_entity_chunks_threads_state_and_strips_quotes() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.query_entity_chunks(["O'Melanie", "war"], state="consolidated")
    cypher = client.query.await_args_list[0].args[1]
    # sanitize() strips quotes; long names use CONTAINS on name_norm, short exact.
    assert "e.name_norm CONTAINS 'omelanie'" in cypher
    assert "e.name_norm = 'war'" in cypher
    assert "c.state = 'consolidated'" in cypher


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
async def test_query_frame_chunks_traverses_evoked_by() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.query_frame_chunks(["Motion"], session_id="s1")
    cypher = client.query.await_args_list[0].args[1]
    assert "(n:FRAME)-[:EVOKED_BY]->(c:CHUNK)" in cypher
    assert "n.frame_type IN ['Motion']" in cypher
    assert "n.session_id = 's1'" in cypher


@pytest.mark.asyncio
async def test_entity_relation_graph_excludes_structural_types() -> None:
    client = _mock_client(
        query_rows=[
            {"src": "Melanie", "type": "LIKES", "dst": "Camping",
             "weight": 0.8, "is_current": True},
        ]
    )
    store = GraphStore(client=client)
    rows = await store.entity_relation_graph()
    assert rows == [
        {"src": "Melanie", "type": "LIKES", "dst": "Camping", "weight": 0.8, "is_current": True}
    ]
    cypher = client.query.await_args_list[0].args[1]
    assert "NOT type(r) IN [" in cypher
    assert "'MENTIONS'" in cypher


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
        c for c in client.command.await_args_list if "MERGE (c:CHUNK {chunk_id: row.cid})" in c.args[1]
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
async def test_write_chunk_embedding_is_inline_literal() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    await store.write_chunk(
        {"chunk_id": "c1", "session_id": "s1", "file_id": "f1", "text": "t",
         "embedding": [0.5, -0.25]}
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
                {"t": "LIKES", "dst": "Camping", "dst_nn": "camping",
                 "predicate": "LIKES", "fact": "", "evidence": ["c1"],
                 "confidence": 0.8, "valid_at": None, "source": "gliner"},
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
    # Relation re-created on the winner via write_relation (LIKES edge).
    merge_rel = next(s for s in stmts if "MERGE (a)-[r:LIKES]->(b)" in s)
    rel_call = next(
        c for c in client.command.await_args_list if "MERGE (a)-[r:LIKES]->(b)" in c.args[1]
    )
    assert rel_call.kwargs["params"]["src_nn"] == "melanie"
    assert rel_call.kwargs["params"]["dst_nn"] == "camping"
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
    assert any(
        "REBUILD GRAPH ANALYTICAL VIEW" in c.args[1] for c in client.command.await_args_list
    )


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
    set_calls = [c for c in client.command.await_args_list if "e.pagerank = $pr" in c.args[1]]
    assert len(set_calls) == 2
    prs = sorted(c.kwargs["params"]["pr"] for c in set_calls)
    assert prs == [0.15, 0.42]


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_falls_back_to_numpy_on_native_failure() -> None:
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "algo.pageRank" in cypher:
            raise Exception("CommandSQLParsingException: algo.pageRank unsupported")
        if "RETURN e.name_norm AS nn" in cypher:
            return [{"nn": "melanie"}, {"nn": "caroline"}]
        if "RETURN DISTINCT a.name_norm AS src" in cypher:
            return [{"src": "melanie", "dst": "caroline", "w": 1.0}]
        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    store = GraphStore(client=client)
    updated = await store.compute_and_persist_pagerank()
    assert updated == 2  # numpy fallback still wrote both entities
    set_calls = [c for c in client.command.await_args_list if "e.pagerank = $pr" in c.args[1]]
    assert len(set_calls) == 2


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_falls_back_when_all_ranks_null() -> None:
    """Regression guard: on a real ~190-entity ArcadeDB 26.5.1 graph,
    ``CALL algo.pageRank()`` returns rows with every ``rank`` null (verified
    live during WS4) — that must be treated as a failure, not persisted as
    meaningless zeros.
    """
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "algo.pageRank" in cypher:
            return [{"nn": "melanie", "rank": None}, {"nn": "caroline", "rank": None}]
        if "RETURN e.name_norm AS nn" in cypher:
            return [{"nn": "melanie"}, {"nn": "caroline"}]
        if "RETURN DISTINCT a.name_norm AS src" in cypher:
            return [{"src": "melanie", "dst": "caroline", "w": 1.0}]
        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    store = GraphStore(client=client)
    updated = await store.compute_and_persist_pagerank()
    assert updated == 2  # fell back to numpy, still wrote both entities
    set_calls = [c for c in client.command.await_args_list if "e.pagerank = $pr" in c.args[1]]
    assert len(set_calls) == 2


@pytest.mark.asyncio
async def test_compute_and_persist_pagerank_skip_view_ensure_avoids_rebuild() -> None:
    client = _mock_client(query_rows=[{"nn": "melanie", "rank": 0.5}])
    store = GraphStore(client=client)
    await store.compute_and_persist_pagerank(skip_view_ensure=True)
    assert not any(
        "GRAPH ANALYTICAL VIEW" in c.args[1] for c in client.command.await_args_list
    )


@pytest.mark.asyncio
async def test_compute_graph_embeddings_writes_list_property() -> None:
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "algo.node2vec" in cypher:
            return [{"nn": "melanie", "embedding": [0.1, 0.2, 0.3]}]
        if "(a:ENTITY)-[r]->(b:ENTITY)" in cypher:
            return [{"nn": "melanie", "deg": 2}]
        if "(a:ENTITY)<-[r]-(b:ENTITY)" in cypher:
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
    assert not any(
        "e.graph_embedding = [" in c.args[1] for c in client.command.await_args_list
    )


@pytest.mark.asyncio
async def test_compute_graph_embeddings_soft_fails_on_native_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=Exception("algo.node2vec not supported"))
    store = GraphStore(client=client)
    updated = await store.compute_graph_embeddings()
    assert updated == 0
    assert not any(
        "e.graph_embedding = [" in c.args[1] for c in client.command.await_args_list
    )


@pytest.mark.asyncio
async def test_compute_communities_writes_community_id() -> None:
    client = _mock_client(query_rows=[{"nn": "melanie", "cid": 2}, {"nn": "caroline", "cid": 2}])
    store = GraphStore(client=client)
    updated = await store.compute_communities()
    assert updated == 2
    set_calls = [c for c in client.command.await_args_list if "e.community_id = $cid" in c.args[1]]
    assert len(set_calls) == 2
    assert all(c.kwargs["params"]["cid"] == 2 for c in set_calls)


@pytest.mark.asyncio
async def test_compute_communities_soft_fails_to_zero_on_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=Exception("algo.louvain not supported"))
    store = GraphStore(client=client)
    updated = await store.compute_communities()
    assert updated == 0
    assert not any(
        "e.community_id = $cid" in c.args[1] for c in client.command.await_args_list
    )


@pytest.mark.asyncio
async def test_compute_communities_soft_fails_on_empty_result() -> None:
    client = _mock_client(query_rows=[])
    store = GraphStore(client=client)
    assert await store.compute_communities() == 0
    assert not any(
        "e.community_id = $cid" in c.args[1] for c in client.command.await_args_list
    )


@pytest.mark.asyncio
async def test_chunk_community_weights_majority_and_size() -> None:
    client = _mock_client()

    async def query_side_effect(db, cypher, params=None):
        if "MENTIONS]->(e:ENTITY)" in cypher:
            # Pre-sorted by cnt desc, as ArcadeDB's ORDER BY would return.
            return [
                {"chunk_id": "c1", "cid": 2, "cnt": 3},
                {"chunk_id": "c2", "cid": 1, "cnt": 2},
                {"chunk_id": "c1", "cid": 1, "cnt": 1},
            ]
        if "count(e) AS size" in cypher:
            return [{"cid": 1, "size": 5}, {"cid": 2, "size": 8}]
        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    store = GraphStore(client=client)
    weights = await store.chunk_community_weights(["c1", "c2"])
    assert weights == {"c1": 8.0, "c2": 5.0}


@pytest.mark.asyncio
async def test_chunk_community_weights_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    assert await store.chunk_community_weights([]) == {}
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunk_community_weights_soft_fails_on_error() -> None:
    client = _mock_client()
    client.query = AsyncMock(side_effect=Exception("boom"))
    store = GraphStore(client=client)
    assert await store.chunk_community_weights(["c1"]) == {}


@pytest.mark.asyncio
async def test_similar_entities_by_embedding_cosine_ranks() -> None:
    client = _mock_client(
        query_rows=[
            {"nn": "melanie", "name": "Melanie", "emb": [1.0, 0.0]},
            {"nn": "mel", "name": "Mel", "emb": [1.0, 0.0]},  # identical -> sim 1.0
            {"nn": "camping", "name": "Camping", "emb": [0.0, 1.0]},  # orthogonal -> sim 0.0
        ]
    )
    store = GraphStore(client=client)
    out = await store.similar_entities_by_embedding("Melanie", top_k=5)
    assert [r["name_norm"] for r in out] == ["mel", "camping"]
    assert out[0]["similarity"] == pytest.approx(1.0)
    assert out[1]["similarity"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_similar_entities_by_embedding_empty_when_seed_missing() -> None:
    client = _mock_client(query_rows=[{"nn": "other", "name": "Other", "emb": [1.0, 0.0]}])
    store = GraphStore(client=client)
    assert await store.similar_entities_by_embedding("Melanie") == []


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
        await store.ensure_file("file-1", "sess-1", name="conv.txt", media_type="text/plain")

        emb_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        await store.write_chunks_bulk(
            [
                {"chunk_id": "chunk-a", "session_id": "sess-1", "file_id": "file-1",
                 "text": "Melanie went camping in Banff.", "seq": 0, "embedding": emb_a},
                {"chunk_id": "chunk-b", "session_id": "sess-1", "file_id": "file-1",
                 "text": "Caroline moved to Toronto.", "seq": 1, "embedding": emb_b},
            ]
        )
        await store.write_entities_bulk(
            "sess-1", "chunk-a",
            [{"name": "Melanie", "type": "PERSON", "confidence": 0.9},
             {"name": "Banff", "type": "PLACE", "confidence": 0.8}],
        )
        # Same relation from two chunks → evidence list must accumulate both.
        await store.write_relation(
            "Melanie", "LIKES", "Camping", fact="Melanie likes camping",
            chunk_id="chunk-a", confidence=0.7, source="gliner",
        )
        await store.write_relation(
            "Melanie", "LIKES", "Camping", chunk_id="chunk-b", confidence=0.9,
        )

        # ANN read-back
        ann = await store.search_chunks_ann(emb_a, top_k=2, session_id="sess-1", state="raw")
        assert ann and ann[0]["chunk_id"] == "chunk-a"

        # Entity traversal
        ent = await store.query_entity_chunks(["Melanie"], session_id="sess-1", top_k=5)
        assert any(r["chunk_id"] == "chunk-a" for r in ent)

        # Relation graph (structural edges excluded, semantic edge present)
        graph = await store.entity_relation_graph()
        assert any(r["type"] == "LIKES" and r["src"] == "Melanie" for r in graph)

        # Evidence provenance accumulated across both writes; confidence keeps max.
        rows = await store._qry(
            "MATCH (a:ENTITY {name_norm: 'melanie'})-[r:LIKES]->(b) "
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
