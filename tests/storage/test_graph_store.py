"""Unit + integration tests for the single-database GraphStore (golden layer).

Unit tests mock ArcadeDBClient and assert the SQL/Cypher shapes the retriever
depends on: every read speaks the golden schema, threads its source scope,
escapes what it interpolates, and composes the tombstone filter. The
integration test runs only when ArcadeDB is reachable and exercises the full
write→read round trip on a scratch ``mem_test_<uuid>`` database.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.exceptions import StoreError
from graphknows.storage.arcadedb.graph_store import _VERTEX_TYPES, GraphStore
from graphknows.storage.namespace import db_name


def _mock_client(query_rows: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=query_rows or [])
    client.create_database = AsyncMock()
    client.database_exists = AsyncMock(return_value=False)
    client.drop_database = AsyncMock()
    return client


def _command_strings(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


def _last_query(client: MagicMock) -> str:
    return str(client.query.await_args.args[1])


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


async def test_ensure_schema_raises_on_real_error() -> None:
    client = _mock_client()
    client.command = AsyncMock(side_effect=Exception("connection refused"))
    store = GraphStore(client=client)
    with pytest.raises(Exception, match="connection refused"):
        await store.ensure_schema(dims=8)


# ---------------------------------------------------------------------------
# Segment reads
# ---------------------------------------------------------------------------


async def test_search_segments_ann_filters_by_source_over_an_overfetched_pool() -> None:
    rows = [
        {"@rid": "#1:0", "id": "g1", "text": "a", "source_id": "s1", "distance": 0.1},
        {"@rid": "#1:1", "id": "g2", "text": "b", "source_id": "s2", "distance": 0.2},
        {"@rid": "#1:0", "id": "g1", "text": "a", "source_id": "s1", "distance": 0.1},
    ]
    store = GraphStore(client=_mock_client(rows))

    out = await store.search_segments_ann([0.1] * 8, top_k=10, source_ids=["s1"])

    sql = _last_query(store.client)
    assert 'vectorNeighbors("SEGMENT[embedding]"' in sql and ", 200)" in sql
    assert [r["id"] for r in out] == ["g1"]  # scoped, and deduplicated by @rid
    assert out[0]["cosine_score"] == pytest.approx(0.9)


async def test_full_text_search_is_case_folded_sql_ordered_by_id() -> None:
    store = GraphStore(client=_mock_client())

    await store.full_text_search_segments(["camping", "O'Brien"], top_k=7, source_ids=["s1"])

    sql = _last_query(store.client)
    assert sql.startswith("SELECT id, text, role, observed_at, source_id FROM SEGMENT")
    assert "text ILIKE '%camping%' OR text ILIKE '%O\\'Brien%'" in sql
    assert "source_id IN ['s1']" in sql
    assert sql.endswith("ORDER BY id LIMIT 7")


async def test_get_segments_by_ids_keys_rows_by_id() -> None:
    rows = [{"id": "g1", "text": "a", "role": "Gina", "observed_at": "2023-01-20"}]
    store = GraphStore(client=_mock_client(rows))

    out = await store.get_segments_by_ids(["g1", ""])

    assert "g.id IN ['g1']" in _last_query(store.client)
    assert out["g1"]["role"] == "Gina"
    assert await store.get_segments_by_ids([]) == {}


async def test_neighbor_segments_walks_next_one_hop_per_radius_and_credits_the_seed() -> None:
    hops = [
        [{"seed": "g2", "id": "g1", "text": "start", "byte_start": 0}],
        [{"seed": "g1", "id": "g0", "text": "earlier", "byte_start": -5}],
    ]
    client = _mock_client()
    client.query = AsyncMock(side_effect=hops)
    store = GraphStore(client=client)

    out = await store.neighbor_segments(["g2"], radius=2)

    assert [r["id"] for r in out["g2"]] == ["g1", "g0"]
    first, second = (call.args[1] for call in client.query.await_args_list)
    assert "(g:SEGMENT)-[:NEXT]-(n:SEGMENT)" in first and "['g2']" in first
    assert "['g1']" in second
    assert await store.neighbor_segments(["g2"], radius=0) == {}


async def test_segments_mentioning_matches_name_norm_on_word_boundaries() -> None:
    store = GraphStore(client=_mock_client())

    await store.segments_mentioning(["O'Melanie", "war", "of"], top_k=5, source_ids=["s1"])

    q = _last_query(store.client)
    assert "(g:SEGMENT)-[:MENTIONS]->(e:ENTITY)" in q
    assert "e.name_norm = 'o\\'melanie'" in q and "e.name_norm = 'war'" in q
    assert "'of'" not in q  # too short to be a name key
    assert "e.state <> 'forgotten'" in q
    assert "g.source_id IN ['s1']" in q
    assert q.endswith("ORDER BY id LIMIT $k")


async def test_segments_evoking_searches_concepts_then_follows_evokes() -> None:
    concepts = [{"@rid": "#2:0", "uri": "c:place", "label": "Place", "distance": 0.2}]
    evoked = [{"id": "g1", "uri": "c:place", "confidence": 0.5}]
    client = _mock_client()
    client.query = AsyncMock(side_effect=[concepts, evoked])
    store = GraphStore(client=client)

    out = await store.segments_evoking([0.1, 0.2], concept_top_k=5, source_ids=None)

    ann, hop = (call.args[1] for call in client.query.await_args_list)
    assert 'vectorNeighbors("CONCEPT[embedding]"' in ann
    assert "(g:SEGMENT)-[v:EVOKES]->(c:CONCEPT)" in hop and "c.uri IN ['c:place']" in hop
    assert out == [{"id": "g1", "uri": "c:place", "label": "Place", "score": pytest.approx(0.4)}]


# ---------------------------------------------------------------------------
# Entity and fact reads
# ---------------------------------------------------------------------------


async def test_resolve_query_entities_ranks_exact_then_shortest_before_the_cap() -> None:
    rows = [
        {"id": "e3", "name": "Jon's studio", "nn": "jon's studio"},
        {"id": "e1", "name": "Jon", "nn": "jon"},
        {"id": "e2", "name": "Dance studio", "nn": "dance studio"},
    ]
    store = GraphStore(client=_mock_client(rows))

    out = await store.resolve_query_entities(["jon", "studio", "of"], cap=2)

    assert out == [{"id": "e1", "name": "Jon"}, {"id": "e2", "name": "Dance studio"}]
    q = _last_query(store.client)
    assert "e.name_norm STARTS WITH 'studio '" in q and "e.state <> 'forgotten'" in q


async def test_facts_by_entity_reads_both_directions_with_evidence() -> None:
    store = GraphStore(client=_mock_client())

    await store.facts_by_entity(["e1"], limit=3)

    q = _last_query(store.client)
    assert "(s:ENTITY)-[:SUBJECT_OF]->(f:FACT)-[:OBJECT_IS]->(o:ENTITY)" in q
    assert "(s.id IN ['e1'] OR o.id IN ['e1'])" in q
    assert "f.is_current = true" in q and "(f)-[:ASSERTED_IN]->(g:SEGMENT)" in q
    assert q.endswith("ORDER BY confidence DESC, id LIMIT $k")
    assert store.client.query.await_args.kwargs["params"] == {"k": 3}
    assert await store.facts_by_entity([]) == []


async def test_facts_by_predicate_matches_id_or_canonical_and_scopes_evidence() -> None:
    store = GraphStore(client=_mock_client())

    await store.facts_by_predicate(["Works_at", " "], limit=5, source_ids=["s1"])

    q = _last_query(store.client)
    assert "(f)-[:USES]->(p:PREDICATE)" in q
    assert "(p.id IN ['works_at'] OR p.canonical IN ['works_at'])" in q
    assert "g.source_id IN ['s1']" in q
    assert await store.facts_by_predicate([]) == []


async def test_sources_filters_by_uri_only_when_asked() -> None:
    store = GraphStore(client=_mock_client())

    await store.sources()
    assert "WHERE" not in _last_query(store.client)

    await store.sources(uri="session:s1")
    assert "WHERE s.uri = 'session:s1'" in _last_query(store.client)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


async def test_delete_source_removes_facts_then_segments_then_the_source() -> None:
    store = GraphStore(client=_mock_client())

    await store.delete_source("session:s1")

    stmts = _command_strings(store.client)
    assert [s.split("DETACH DELETE ")[1] for s in stmts] == ["f", "g", "s"]
    assert all("SOURCE {uri: $uri}" in s for s in stmts)


async def test_delete_source_reports_a_real_failure() -> None:
    client = _mock_client()
    client.command = AsyncMock(side_effect=[[], Exception("disk full"), []])
    with pytest.raises(StoreError, match="SEGMENT: disk full"):
        await GraphStore(client=client).delete_source("session:s1")


async def test_clear_all_wipes_every_golden_vertex_type_but_the_stamp() -> None:
    store = GraphStore(client=_mock_client())

    await store.clear_all()

    labels = [s.split(":")[1].split(")")[0] for s in _command_strings(store.client)]
    assert labels == list(_VERTEX_TYPES)
    assert "SCHEMA_STAMP" not in labels and {"SOURCE", "SEGMENT", "FACT"} <= set(labels)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_graph_store_round_trip_on_scratch_database(arcadedb_required: None) -> None:
    """Full write→read round trip on a scratch mem_test_<uuid> database.

    SOURCE, ENTITY and CONCEPT are written with raw Cypher here: their writers
    pass ``meta`` / ``type_histogram`` as MAP params and ``embedding`` as a
    LIST param, both of which ArcadeDB's Cypher rejects (a T020 writer defect,
    not a read defect). The reads under test do not care who wrote the row.
    """
    from graphknows.models.fact import Fact, Mention
    from graphknows.models.segment import Segment, SegmentKind
    from graphknows.models.symbols import PredicateRef
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client
    from graphknows.storage.arcadedb._sql import vector_literal
    from graphknows.storage.arcadedb.writers import (
        EntityWriter,
        Evidence,
        FactWriter,
        SegmentWriter,
        SymbolWriter,
    )

    client = build_arcadedb_client(GraphKnowsSettings())
    store = GraphStore(client=client, db=f"mem_test_{uuid.uuid4().hex[:8]}")
    await store.connect()
    try:
        await store.ensure_schema(dims=4)
        await store.ensure_schema(dims=4)  # idempotent second run

        await store.command(
            "MERGE (s:SOURCE {id: 'src-1'}) SET s.uri = 'session:s1', s.content_hash = 'abc', "
            "s.imported_at = '2026-01-01T00:00:00+00:00'"
        )
        first = Segment(
            source_id="src-1",
            text="Melanie went camping in Yosemite.",
            kind=SegmentKind.turn,
            byte_range=(0, 33),
            role="Melanie",
        )
        second = Segment(
            source_id="src-1", text="It rained.", kind=SegmentKind.turn, byte_range=(34, 44)
        )
        for segment in (first, second):
            await SegmentWriter(store).write(segment)
        await store.command(
            "MATCH (a:SEGMENT {id: $a}), (b:SEGMENT {id: $b}) MERGE (a)-[:NEXT]->(b)",
            a=first.id,
            b=second.id,
        )
        await store.command(
            f"MATCH (g:SEGMENT {{id: $id}}) SET g.embedding = {vector_literal([1.0, 0.0, 0.0, 0.0])}",
            id=first.id,
        )
        for entity_id, name in (("e1", "Melanie"), ("e2", "Yosemite")):
            await store.command(
                "MERGE (e:ENTITY {id: $id}) "
                "SET e.name = $name, e.name_norm = $nn, e.state = 'active'",
                id=entity_id,
                name=name,
                nn=name.casefold(),
            )
        await EntityWriter(store).write_mention(
            Mention(segment_id=first.id, entity_id="e1", surface="Melanie", span=(0, 7))
        )
        await store.command(
            "MERGE (c:CONCEPT {uri: 'c:place'}) SET c.label = 'Place', "
            f"c.embedding = {vector_literal([0.0, 1.0, 0.0, 0.0])}"
        )
        symbols = SymbolWriter(store)
        await symbols.write_evokes(first.id, "c:place", 0.8)
        await symbols.write_predicate(
            PredicateRef(
                id="p:went", label="went to", definition="moved to", canonical="went", pack="t"
            )
        )
        fact = Fact(id="f1", subject="e1", predicate="p:went", object="e2", confidence=0.9)
        await FactWriter(store).write(fact, [Evidence(segment_id=first.id, span=(0, 33))])

        [source_row] = await store.sources(uri="session:s1")
        assert source_row["id"] == "src-1"

        ann = await store.search_segments_ann([1.0, 0.0, 0.0, 0.0], 2, ["src-1"])
        assert [r["id"] for r in ann] == [first.id]
        assert ann[0]["role"] == "Melanie"

        pool = await store.full_text_search_segments(["CAMPING"], 5, ["src-1"])
        assert [r["id"] for r in pool] == [first.id]

        by_id = await store.get_segments_by_ids([first.id, second.id])
        assert set(by_id) == {first.id, second.id}

        neighbours = await store.neighbor_segments([first.id], radius=1)
        assert [n["id"] for n in neighbours[first.id]] == [second.id]

        mentioning = await store.segments_mentioning(["melanie"], 5, ["src-1"])
        assert [r["id"] for r in mentioning] == [first.id]

        evoking = await store.segments_evoking([0.0, 1.0, 0.0, 0.0], 3, ["src-1"])
        assert [(r["id"], r["uri"]) for r in evoking] == [(first.id, "c:place")]

        assert await store.resolve_query_entities(["melanie"]) == [{"id": "e1", "name": "Melanie"}]

        [by_entity] = await store.facts_by_entity(["e2"])
        assert (by_entity["subject_name"], by_entity["object_name"]) == ("Melanie", "Yosemite")
        assert by_entity["segment_id"] == first.id

        [by_predicate] = await store.facts_by_predicate(["went"], source_ids=["src-1"])
        assert by_predicate["id"] == "f1"

        assert store.read_types >= {"SOURCE", "SEGMENT", "NEXT", "MENTIONS", "EVOKES", "FACT"}

        await store.delete_source("session:s1")
        assert await store.sources() == []
        assert await store.get_segments_by_ids([first.id]) == {}
        assert await store.facts_by_entity(["e1"]) == []
    finally:
        await store.drop_database()
        await store.close()
