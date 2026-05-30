"""Unit tests for Neo4jStore and neo4j adapter helpers."""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from graphknows.retrieval.stores.neo4j import (
    Neo4jStore,
    _GK_NS,
    _safe_rel_label,
    _write_evidence_graph,
    _write_relation_edges,
    chunk_node_id,
    entity_node_id,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _mock_session() -> MagicMock:
    s = MagicMock()
    s.run = MagicMock(return_value=None)
    return s


def _stm_with_entities() -> dict:
    return {
        "files": [{"id": "f1", "title": "T", "parser_name": "plain", "source_hint": ""}],
        "chunks": [
            {
                "id": "c1",
                "text": "EU regulates ECB.",
                "heading_path": "",
                "chunk_index": 0,
                "file_id": "f1",
            }
        ],
        "chunk_entity_edges": [
            {"chunk_id": "c1", "entity_name": "EU"},
            {"chunk_id": "c1", "entity_name": "ECB"},
        ],
        "entities": [
            {"name": "EU", "type": "ORG", "confidence": 0.95},
            {"name": "ECB", "type": "ORG", "confidence": 0.92},
        ],
        "dialogs": [],
        "turns": [],
        "turn_entity_edges": [],
        "next_edges": [],
    }


def _make_store():
    """Build a Neo4jStore with mocked neo4j driver — no DB connection needed."""
    store = Neo4jStore.__new__(Neo4jStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = iter([])
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    store._db = "neo4j"
    return store, mock_driver, mock_session


def _make_neo4j_store():
    """Return a Neo4jStore with a mocked driver (no real DB)."""
    store = Neo4jStore.__new__(Neo4jStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    store._db = "neo4j"
    return store, mock_driver, mock_session


def _make_run_result(rows: list[dict]):
    """Build a mock that looks like a neo4j result with .data()."""
    mock = MagicMock()
    mock.data.return_value = rows
    mock.__iter__ = lambda self: iter(rows)
    return mock


# ---------------------------------------------------------------------------
# _safe_rel_label (merged from test_neo4j_store.py and test_neo4j_store_extended.py)
# ---------------------------------------------------------------------------


class TestSafeRelLabel:
    def test_simple_uppercase(self) -> None:
        assert _safe_rel_label("regulates") == "REGULATES"

    def test_spaces_become_underscores(self) -> None:
        assert _safe_rel_label("is funded by") == "IS_FUNDED_BY"

    def test_hyphens_become_underscores(self) -> None:
        assert _safe_rel_label("co-sponsors") == "CO_SPONSORS"

    def test_strips_special_chars(self) -> None:
        assert _safe_rel_label("regulates!") == "REGULATES"

    def test_empty_string(self) -> None:
        assert _safe_rel_label("") == ""

    def test_lowercased_input_uppercased(self):
        assert _safe_rel_label("works at") == "WORKS_AT"

    def test_hyphens_to_underscores(self):
        assert _safe_rel_label("co-author") == "CO_AUTHOR"

    def test_already_valid(self):
        assert _safe_rel_label("REGULATES") == "REGULATES"

    def test_strips_non_word_chars(self):
        result = _safe_rel_label("foo!bar@baz")
        assert result == "FOOBARBAZ"

    def test_spaces_to_underscores(self):
        assert _safe_rel_label("part of") == "PART_OF"

    def test_only_special_chars(self):
        assert _safe_rel_label("!!!") == ""


# ---------------------------------------------------------------------------
# chunk_node_id
# ---------------------------------------------------------------------------


class TestChunkNodeId:
    def test_returns_uuid_string(self):
        result = chunk_node_id("doc1", "hashABC")
        uuid.UUID(result)

    def test_deterministic(self):
        assert chunk_node_id("doc1", "hash") == chunk_node_id("doc1", "hash")

    def test_different_inputs_differ(self):
        assert chunk_node_id("doc1", "hash1") != chunk_node_id("doc1", "hash2")
        assert chunk_node_id("doc1", "hash") != chunk_node_id("doc2", "hash")

    def test_uses_gk_namespace(self):
        expected = str(uuid.uuid5(_GK_NS, "chunk:doc1:hash"))
        assert chunk_node_id("doc1", "hash") == expected


# ---------------------------------------------------------------------------
# entity_node_id
# ---------------------------------------------------------------------------


class TestEntityNodeId:
    def test_returns_uuid_string(self):
        result = entity_node_id("Alice")
        uuid.UUID(result)

    def test_case_insensitive(self):
        assert entity_node_id("alice") == entity_node_id("ALICE") == entity_node_id("Alice")

    def test_strips_whitespace(self):
        assert entity_node_id("  Alice  ") == entity_node_id("Alice")

    def test_different_names_differ(self):
        assert entity_node_id("Alice") != entity_node_id("Bob")


# ---------------------------------------------------------------------------
# _write_evidence_graph (merged from test_neo4j_store.py and test_wave3_neo4j_advanced.py)
# ---------------------------------------------------------------------------


class TestWriteEvidenceGraph:
    def test_no_co_occurs(self) -> None:
        """CO_OCCURS must never appear in _write_evidence_graph queries."""
        s = _mock_session()
        stm = _stm_with_entities()
        entity_by_name = {e["name"]: e for e in stm["entities"]}
        _write_evidence_graph(s, "sess-1", stm, entity_by_name, "2026-01-01")
        calls = " ".join(str(c) for c in s.run.call_args_list)
        assert "CO_OCCURS" not in calls

    def test_has_entity_present(self) -> None:
        s = _mock_session()
        stm = _stm_with_entities()
        entity_by_name = {e["name"]: e for e in stm["entities"]}
        _write_evidence_graph(s, "sess-1", stm, entity_by_name, "2026-01-01")
        calls = " ".join(str(c) for c in s.run.call_args_list)
        assert "HAS_ENTITY" in calls

    def test_returns_chunk_to_ents_map(self) -> None:
        s = _mock_session()
        stm = _stm_with_entities()
        entity_by_name = {e["name"]: e for e in stm["entities"]}
        nodes, edges, c2e = _write_evidence_graph(s, "sess-1", stm, entity_by_name, "2026-01-01")
        assert "c1" in c2e
        assert "EU" in c2e["c1"]
        assert "ECB" in c2e["c1"]
        assert isinstance(nodes, int) and isinstance(edges, int)

    def _call(self, stm_data: dict):
        mock_session = MagicMock()
        nodes, edges, chunk_to_ents = _write_evidence_graph(
            s=mock_session,
            session_id="sess-1",
            stm_data=stm_data,
            entity_by_name={
                "ALICE": {"type": "PERSON", "confidence": 0.9},
            },
            now="2024-01-01T00:00:00",
        )
        return mock_session, nodes, edges, chunk_to_ents

    def test_empty_stm_data_writes_session_only(self):
        session, nodes, edges, chunk_to_ents = self._call({})
        assert any(
            "MERGE" in str(c) and "SESSION" in str(c)
            for c in [str(call) for call in session.run.call_args_list]
        )
        assert nodes >= 1
        assert edges == 0
        assert chunk_to_ents == {}

    def test_file_nodes_increment_counts(self):
        stm = {
            "files": [{"id": "f1", "title": "Doc 1", "parser_name": "text", "source_hint": "web"}]
        }
        session, nodes, edges, chunk_to_ents = self._call(stm)
        assert nodes >= 2
        assert edges >= 1

    def test_chunk_nodes_written(self):
        stm = {
            "files": [{"id": "f1", "title": "T", "parser_name": "p", "source_hint": ""}],
            "chunks": [
                {
                    "id": "c1",
                    "text": "hello",
                    "heading_path": "/h",
                    "chunk_index": 0,
                    "file_id": "f1",
                }
            ],
        }
        session, nodes, edges, _ = self._call(stm)
        cypher_calls = [str(call) for call in session.run.call_args_list]
        assert any("CHUNK" in c for c in cypher_calls)

    def test_entity_edges_populate_chunk_to_ents(self):
        stm = {
            "chunk_entity_edges": [{"entity_name": "ALICE", "chunk_id": "c1"}],
        }
        session, nodes, edges, chunk_to_ents = self._call(stm)
        assert "c1" in chunk_to_ents
        assert "ALICE" in chunk_to_ents["c1"]

    def test_entity_only_written_once(self):
        stm = {
            "chunk_entity_edges": [
                {"entity_name": "ALICE", "chunk_id": "c1"},
                {"entity_name": "ALICE", "chunk_id": "c2"},
            ]
        }
        session, _, _, chunk_to_ents = self._call(stm)
        merge_entity_calls = [
            str(c) for c in session.run.call_args_list if "MERGE (e:ENTITY" in str(c)
        ]
        assert len(merge_entity_calls) == 1


# ---------------------------------------------------------------------------
# _write_relation_edges (merged from test_neo4j_store.py and test_neo4j_store_extended.py)
# ---------------------------------------------------------------------------


class TestWriteRelationEdges:
    def _make_mock_session(self):
        s = MagicMock()
        s.run = MagicMock(return_value=MagicMock())
        return s

    def test_dynamic_label_used(self) -> None:
        s = _mock_session()
        eb = {"EU": {"type": "ORG"}, "ECB": {"type": "ORG"}}
        triplets = [("EU", "REGULATES", "ECB")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 1
        call_str = " ".join(str(c) for c in s.run.call_args_list)
        assert "REGULATES" in call_str

    def test_multiple_relation_types(self) -> None:
        s = _mock_session()
        eb = {"EU": {}, "ECB": {}, "EIB": {}}
        triplets = [("EU", "REGULATES", "ECB"), ("EU", "FUNDS", "EIB")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 2
        call_str = " ".join(str(c) for c in s.run.call_args_list)
        assert "REGULATES" in call_str
        assert "FUNDS" in call_str

    def test_unknown_endpoint_skipped(self) -> None:
        s = _mock_session()
        eb = {"EU": {}}
        triplets = [("EU", "REGULATES", "GHOST")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 0
        s.run.assert_not_called()

    def test_empty_relation_label_skipped(self) -> None:
        s = _mock_session()
        eb = {"EU": {}, "ECB": {}}
        triplets = [("EU", "!@#", "ECB")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 0
        s.run.assert_not_called()

    def test_lowercases_normalised(self) -> None:
        s = _mock_session()
        eb = {"EU": {}, "ECB": {}}
        triplets = [("EU", "funds", "ECB")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 1
        call_str = str(s.run.call_args_list[0])
        assert "FUNDS" in call_str

    def test_writes_valid_edges(self):
        s = self._make_mock_session()
        entity_by_name = {
            "Alice": {"type": "PERSON"},
            "Bob": {"type": "PERSON"},
        }
        count = _write_relation_edges(s, [("Alice", "KNOWS", "Bob")], entity_by_name)
        assert count == 1
        s.run.assert_called_once()
        cypher = s.run.call_args[0][0]
        assert "KNOWS" in cypher
        assert "MERGE" in cypher

    def test_skips_unknown_endpoints(self):
        s = self._make_mock_session()
        entity_by_name = {"Alice": {}}
        count = _write_relation_edges(s, [("Alice", "KNOWS", "Unknown")], entity_by_name)
        assert count == 0
        s.run.assert_not_called()

    def test_skips_empty_relation_label(self):
        s = self._make_mock_session()
        entity_by_name = {"Alice": {}, "Bob": {}}
        count = _write_relation_edges(s, [("Alice", "!!!", "Bob")], entity_by_name)
        assert count == 0

    def test_multiple_edges(self):
        s = self._make_mock_session()
        entity_by_name = {"A": {}, "B": {}, "C": {}}
        count = _write_relation_edges(
            s,
            [("A", "KNOWS", "B"), ("B", "RELATES_TO", "C"), ("A", "RELATES_TO", "C")],
            entity_by_name,
        )
        assert count == 3
        assert s.run.call_count == 3

    def test_relation_label_normalized(self):
        s = self._make_mock_session()
        entity_by_name = {"A": {}, "B": {}}
        _write_relation_edges(s, [("A", "works for", "B")], entity_by_name)
        cypher = s.run.call_args[0][0]
        assert "WORKS_FOR" in cypher


# ---------------------------------------------------------------------------
# Neo4jStore lifecycle
# ---------------------------------------------------------------------------


class TestNeo4jStoreLifecycle:
    def test_close(self):
        store, mock_driver, _ = _make_store()
        store.close()
        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_indexes_ignores_exceptions(self):
        store, _, mock_session = _make_store()
        mock_session.run.side_effect = Exception("already exists")
        await store.ensure_indexes()


# ---------------------------------------------------------------------------
# Neo4jStore.upsert_chunks
# ---------------------------------------------------------------------------


class TestNeo4jStoreUpsertChunks:
    @pytest.mark.asyncio
    async def test_empty_chunks_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_chunks([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_chunks_cypher(self):
        store, _, mock_session = _make_store()
        await store.upsert_chunks([{"id": "c1", "doc_id": "d1", "text": "hello world"}])
        assert mock_session.run.called
        cypher = mock_session.run.call_args[0][0]
        assert ":CHUNK" in cypher
        assert "MERGE" in cypher

    @pytest.mark.asyncio
    async def test_upsert_chunks_truncates_text(self):
        store, _, mock_session = _make_store()
        long_text = "x" * 4000
        await store.upsert_chunks([{"id": "c1", "doc_id": "d", "text": long_text}])
        kwargs = mock_session.run.call_args[1]
        assert len(kwargs["text"]) <= 3000


# ---------------------------------------------------------------------------
# Neo4jStore.upsert_entities_and_mentions
# ---------------------------------------------------------------------------


class TestNeo4jStoreUpsertEntitiesAndMentions:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_entities_and_mentions([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_entity_and_mentions(self):
        store, _, mock_session = _make_store()
        chunks = [
            {
                "id": "c1",
                "entities": [{"text": "Alice", "label": "PERSON"}],
            }
        ]
        await store.upsert_entities_and_mentions(chunks)
        assert mock_session.run.call_count >= 2
        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        assert any(":ENTITY" in cy for cy in all_cyphers)
        assert any("MENTIONS" in cy for cy in all_cyphers)

    @pytest.mark.asyncio
    async def test_skips_short_entity_names(self):
        store, _, mock_session = _make_store()
        chunks = [{"id": "c1", "entities": [{"text": "a", "label": "X"}]}]
        await store.upsert_entities_and_mentions(chunks)
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_entity_label_fallback_chain(self):
        store, _, mock_session = _make_store()
        chunks = [{"id": "c1", "entities": [{"text": "Bitcoin"}]}]
        await store.upsert_entities_and_mentions(chunks)
        entity_call = mock_session.run.call_args_list[0]
        assert entity_call[1]["label"] == "Entity"


# ---------------------------------------------------------------------------
# Neo4jStore.upsert_topics
# ---------------------------------------------------------------------------


class TestNeo4jStoreUpsertTopics:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_topics([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_topic_node_and_edges(self):
        store, _, mock_session = _make_store()
        await store.upsert_topics(
            [
                {
                    "id": "t1",
                    "label": "AI Safety",
                    "summary": "...",
                    "community_id": 0,
                    "modularity": 0.5,
                    "silhouette": 0.3,
                    "members": ["Alice", "Bob"],
                }
            ]
        )
        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        assert any(":TOPIC" in cy for cy in all_cyphers)
        assert any("IN_COMMUNITY" in cy for cy in all_cyphers)


# ---------------------------------------------------------------------------
# Neo4jStore.upsert_co_occurs
# ---------------------------------------------------------------------------


class TestNeo4jStoreUpsertCoOccurs:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_co_occurs([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_co_occurs_edge(self):
        store, _, mock_session = _make_store()
        await store.upsert_co_occurs([("Alice", "Bob", 3)])
        cypher = mock_session.run.call_args[0][0]
        assert "CO_OCCURS" in cypher
        assert mock_session.run.call_args[1]["w"] == 3


# ---------------------------------------------------------------------------
# Neo4jStore.get_chunks_for_entity
# ---------------------------------------------------------------------------


class TestNeo4jStoreGetChunksForEntity:
    @pytest.mark.asyncio
    async def test_returns_empty_if_no_results(self):
        store, _, mock_session = _make_store()
        mock_session.run.return_value = iter([])
        results = await store.get_chunks_for_entity("Alice")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_parsed_chunk_dicts(self):
        store, _, mock_session = _make_store()
        fake_row = {"chunk_id": "c1", "doc_id": "d1", "text": "hello"}
        mock_session.run.return_value = iter([fake_row])

        results = await store.get_chunks_for_entity("Alice", top_k=3)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        cypher = mock_session.run.call_args[0][0]
        assert "MENTIONS" in cypher or ":ENTITY" in cypher


# ---------------------------------------------------------------------------
# Neo4jStore._arun
# ---------------------------------------------------------------------------


class TestNeo4jStoreArun:
    @pytest.mark.asyncio
    async def test_arun_delegates_to_run(self):
        store, _, mock_session = _make_store()
        mock_session.run.return_value = iter([])
        result = await store._arun("RETURN 1 AS n")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _compute_silhouette (from test_wave3_neo4j_advanced.py)
# ---------------------------------------------------------------------------


class TestComputeSilhouette:
    def test_returns_none_for_single_community(self):
        from graphknows.retrieval.stores.neo4j import _compute_silhouette

        result = _compute_silhouette(
            names=["A", "B"],
            name_idx={"A": 0, "B": 1},
            communities={0: ["A", "B"]},
            rows=[],
            weights=[],
        )
        assert result is None

    def test_returns_none_for_empty_names(self):
        from graphknows.retrieval.stores.neo4j import _compute_silhouette

        result = _compute_silhouette(
            names=[],
            name_idx={},
            communities={},
            rows=[],
            weights=[],
        )
        assert result is None

    def test_returns_float_or_none_for_two_communities(self):
        from graphknows.retrieval.stores.neo4j import _compute_silhouette

        result = _compute_silhouette(
            names=["A", "B"],
            name_idx={"A": 0, "B": 1},
            communities={0: ["A"], 1: ["B"]},
            rows=[{"a": "A", "b": "B"}],
            weights=[1.0],
        )
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# _write_interaction_graph (from test_wave3_neo4j_advanced.py)
# ---------------------------------------------------------------------------


class TestWriteInteractionGraph:
    def _call(self, stm_data: dict):
        from graphknows.retrieval.stores.neo4j import _write_interaction_graph

        mock_session = MagicMock()
        nodes, edges = _write_interaction_graph(
            s=mock_session,
            session_id="sess-1",
            stm_data=stm_data,
            entity_by_name={"BOB": {"type": "PERSON", "confidence": 0.95}},
            base_written=set(),
        )
        return mock_session, nodes, edges

    def test_empty_data_returns_zeros(self):
        _, nodes, edges = self._call({})
        assert nodes == 0
        assert edges == 0

    def test_dialog_creates_node_and_edge(self):
        stm = {
            "dialogs": [{"id": "d1", "started_at": "2024-01-01"}],
        }
        _, nodes, edges = self._call(stm)
        assert nodes >= 1
        assert edges >= 1

    def test_turn_creates_node_and_edge(self):
        stm = {
            "dialogs": [],
            "turns": [
                {
                    "id": "t1",
                    "dialog_id": "d1",
                    "turn_index": 0,
                    "role": "user",
                    "text": "hello",
                }
            ],
        }
        _, nodes, edges = self._call(stm)
        assert nodes >= 1

    def test_next_edge_written(self):
        stm = {
            "next_edges": [{"from_turn_id": "t1", "to_turn_id": "t2"}],
        }
        session, _, edges = self._call(stm)
        assert edges >= 1
        cypher_calls = [str(c) for c in session.run.call_args_list]
        assert any("NEXT" in c for c in cypher_calls)

    def test_turn_entity_edge_written_once_per_entity(self):
        stm = {
            "turn_entity_edges": [
                {"entity_name": "BOB", "turn_id": "t1"},
                {"entity_name": "BOB", "turn_id": "t2"},
            ]
        }
        session, _, _ = self._call(stm)
        merge_entity_calls = [
            str(c) for c in session.run.call_args_list if "MERGE (e:ENTITY" in str(c)
        ]
        assert len(merge_entity_calls) == 1

    def test_base_written_prevents_duplicate_entity_merge(self):
        stm = {
            "turn_entity_edges": [{"entity_name": "BOB", "turn_id": "t1"}],
        }
        from graphknows.retrieval.stores.neo4j import _write_interaction_graph

        mock_session = MagicMock()
        _write_interaction_graph(
            s=mock_session,
            session_id="s1",
            stm_data=stm,
            entity_by_name={},
            base_written={"BOB"},
        )
        merge_calls = [
            str(c) for c in mock_session.run.call_args_list if "MERGE (e:ENTITY" in str(c)
        ]
        assert len(merge_calls) == 0


# ---------------------------------------------------------------------------
# Neo4jStore.merge_session_entities_to_corpus
# ---------------------------------------------------------------------------


class TestMergeSessionEntitiesToCorpus:
    def test_merges_valid_nodes_and_edges(self):
        store, _, _ = _make_neo4j_store()
        store._run = MagicMock(return_value=[])

        nodes = [{"name": "alice", "type": "PERSON"}]
        edges = [{"from": "alice", "to": "bob", "rel_type": "KNOWS"}]

        result = asyncio.run(store.merge_session_entities_to_corpus("s1", nodes, edges))
        merged_nodes, merged_edges = result
        assert merged_nodes == 1
        assert merged_edges == 1

    def test_skips_empty_names(self):
        store, _, _ = _make_neo4j_store()
        store._run = MagicMock(return_value=[])

        nodes = [{"name": "", "type": "PERSON"}, {"name": "   ", "type": "ORG"}]
        result = asyncio.run(store.merge_session_entities_to_corpus("s1", nodes, []))
        merged_nodes, merged_edges = result
        assert merged_nodes == 0

    def test_skips_edges_with_missing_endpoints(self):
        store, _, _ = _make_neo4j_store()
        store._run = MagicMock(return_value=[])

        edges = [{"from": "", "to": "BOB", "rel_type": "KNOWS"}]
        result = asyncio.run(store.merge_session_entities_to_corpus("s1", [], edges))
        _, merged_edges = result
        assert merged_edges == 0


# ---------------------------------------------------------------------------
# Neo4jStore.preview_session_merge
# ---------------------------------------------------------------------------


class TestPreviewSessionMerge:
    def test_returns_expected_keys(self):
        store, _, _ = _make_neo4j_store()

        count_row = {"c": 2}
        store._run = MagicMock(return_value=[count_row])

        nodes = [{"name": "alice"}, {"name": "bob"}]
        result = asyncio.run(store.preview_session_merge("s1", nodes, [{"from": "a", "to": "b"}]))

        assert result["session_id"] == "s1"
        assert result["node_count"] == 2
        assert result["edge_count"] == 1
        assert "existing_overlap_count" in result
        assert "sample_nodes" in result

    def test_empty_nodes_skips_db_query(self):
        store, _, _ = _make_neo4j_store()
        store._run = MagicMock(return_value=[])
        result = asyncio.run(store.preview_session_merge("s1", [], []))
        assert result["node_count"] == 0
        assert result["existing_overlap_count"] == 0


# ---------------------------------------------------------------------------
# Neo4jStore.get_chunk_entities
# ---------------------------------------------------------------------------


class TestGetChunkEntities:
    def test_returns_entities(self):
        store, _, _ = _make_neo4j_store()

        row1 = {"name": "Alice", "label": "PERSON"}
        row2 = {"name": "EU", "label": None}
        store._run = MagicMock(return_value=[row1, row2])

        result = asyncio.run(store.get_chunk_entities("chunk-1"))
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["label"] == "Entity"

    def test_returns_empty_for_unknown_chunk(self):
        store, _, _ = _make_neo4j_store()
        store._run = MagicMock(return_value=[])
        result = asyncio.run(store.get_chunk_entities("nonexistent"))
        assert result == []


# ---------------------------------------------------------------------------
# Neo4jStore.get_entity_neighbors
# ---------------------------------------------------------------------------


class TestGetEntityNeighbors:
    def test_filters_invalid_relationship_types(self):
        store, _, _ = _make_neo4j_store()

        row_data = [{"name": "Bob", "node_label": "PERSON", "rel_path": ["KNOWS"]}]
        store._run = MagicMock(return_value=row_data)

        result = asyncio.run(
            store.get_entity_neighbors(
                "alice",
                depth=2,
                relationship_types=["KNOWS", "DROP TABLE", "VALID_TYPE"],
            )
        )

        assert isinstance(result, dict)

    def test_depth_is_capped_at_3(self):
        store, _, _ = _make_neo4j_store()

        cypher_sent = []

        def capture_run(cypher, **kwargs):
            cypher_sent.append(cypher)
            return []

        store._run = capture_run

        asyncio.run(store.get_entity_neighbors("alice", depth=99))

        assert any("3" in c for c in cypher_sent)


# ---------------------------------------------------------------------------
# _escape_cypher_identifier (from test_wave3_neo4j_adapter.py)
# ---------------------------------------------------------------------------


class TestEscapeCypherIdentifier:
    def test_simple_identifier_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("MyLabel") == "MyLabel"

    def test_underscore_identifier_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("my_label") == "my_label"

    def test_empty_string_returned_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("") == ""

    def test_identifier_with_space_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("My Label")
        assert result == "`My Label`"

    def test_identifier_with_dash_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("my-label")
        assert result == "`my-label`"

    def test_identifier_starting_with_digit_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("1node")
        assert result.startswith("`")
        assert result.endswith("`")

    def test_backtick_in_identifier_is_doubled(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("My`Label")
        assert "``" in result
        assert result.startswith("`")

    def test_special_chars_require_escaping(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("label.with.dots")
        assert result.startswith("`")

    def test_pure_alphanum_no_escaping(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("RELATES_TO")
        assert result == "RELATES_TO"


# ---------------------------------------------------------------------------
# _clean_props_for_neo4j (from test_wave3_neo4j_adapter.py)
# ---------------------------------------------------------------------------


class TestCleanPropsForNeo4j:
    def test_removes_none_values(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"a": None, "b": "value"})
        assert "a" not in result
        assert result["b"] == "value"

    def test_preserves_primitives(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"s": "text", "i": 42, "f": 3.14, "b": True})
        assert result["s"] == "text"
        assert result["i"] == 42
        assert result["f"] == 3.14
        assert result["b"] is True

    def test_homogeneous_list_of_strings(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"tags": ["a", "b", "c"]})
        assert result["tags"] == ["a", "b", "c"]

    def test_homogeneous_list_of_ints(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"counts": [1, 2, 3]})
        assert result["counts"] == [1, 2, 3]

    def test_empty_list_excluded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"empty": []})
        assert "empty" not in result

    def test_mixed_type_list_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"mixed": [1, "two", 3.0]})
        assert isinstance(result["mixed"], str)
        parsed = json.loads(result["mixed"])
        assert parsed[1] == "two"

    def test_dict_value_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"meta": {"key": "value"}})
        assert isinstance(result["meta"], str)
        assert json.loads(result["meta"]) == {"key": "value"}

    def test_complex_object_str_fallback(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        class Unserializable:
            def __repr__(self):
                return "Unserializable()"

        result = _clean_props_for_neo4j({"obj": Unserializable()})
        assert isinstance(result["obj"], str)

    def test_tuple_treated_as_list(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"coords": (1.0, 2.0)})
        assert "coords" in result

    def test_empty_dict_returns_empty(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({})
        assert result == {}

    def test_list_of_dicts_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"items": [{"k": "v"}, {"k": "v2"}]})
        assert isinstance(result["items"], str)

# ---------------------------------------------------------------------------
# _escape_cypher_identifier
# ---------------------------------------------------------------------------


class TestEscapeCypherIdentifier:
    def test_simple_identifier_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("MyLabel") == "MyLabel"

    def test_underscore_identifier_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("my_label") == "my_label"

    def test_empty_string_returned_unchanged(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        assert _escape_cypher_identifier("") == ""

    def test_identifier_with_space_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("My Label")
        assert result == "`My Label`"

    def test_identifier_with_dash_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("my-label")
        assert result == "`my-label`"

    def test_identifier_starting_with_digit_is_escaped(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("1node")
        assert result.startswith("`")
        assert result.endswith("`")

    def test_backtick_in_identifier_is_doubled(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("My`Label")
        # Inner backtick must be doubled: `My``Label`
        assert "``" in result
        assert result.startswith("`")

    def test_special_chars_require_escaping(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("label.with.dots")
        assert result.startswith("`")

    def test_pure_alphanum_no_escaping(self):
        from graphknows.retrieval.stores._cypher_utils import _escape_cypher_identifier

        result = _escape_cypher_identifier("RELATES_TO")
        assert result == "RELATES_TO"


# ---------------------------------------------------------------------------
# _clean_props_for_neo4j
# ---------------------------------------------------------------------------


class TestCleanPropsForNeo4j:
    def test_removes_none_values(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"a": None, "b": "value"})
        assert "a" not in result
        assert result["b"] == "value"

    def test_preserves_primitives(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"s": "text", "i": 42, "f": 3.14, "b": True})
        assert result["s"] == "text"
        assert result["i"] == 42
        assert result["f"] == 3.14
        assert result["b"] is True

    def test_homogeneous_list_of_strings(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"tags": ["a", "b", "c"]})
        assert result["tags"] == ["a", "b", "c"]

    def test_homogeneous_list_of_ints(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"counts": [1, 2, 3]})
        assert result["counts"] == [1, 2, 3]

    def test_empty_list_excluded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"empty": []})
        assert "empty" not in result

    def test_mixed_type_list_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"mixed": [1, "two", 3.0]})
        # Mixed list → JSON string
        assert isinstance(result["mixed"], str)
        parsed = json.loads(result["mixed"])
        assert parsed[1] == "two"

    def test_dict_value_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"meta": {"key": "value"}})
        assert isinstance(result["meta"], str)
        assert json.loads(result["meta"]) == {"key": "value"}

    def test_complex_object_str_fallback(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        # An object that can't be JSON-encoded should fall back to str()
        class Unserializable:
            def __repr__(self):
                return "Unserializable()"

        result = _clean_props_for_neo4j({"obj": Unserializable()})
        assert isinstance(result["obj"], str)

    def test_tuple_treated_as_list(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({"coords": (1.0, 2.0)})
        # Tuples should be handled like lists
        assert "coords" in result

    def test_empty_dict_returns_empty(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        result = _clean_props_for_neo4j({})
        assert result == {}

    def test_list_of_dicts_json_encoded(self):
        from graphknows.retrieval.stores._cypher_utils import _clean_props_for_neo4j

        # List of dicts → first element is dict → complex list → JSON string
        result = _clean_props_for_neo4j({"items": [{"k": "v"}, {"k": "v2"}]})
        assert isinstance(result["items"], str)


# ===========================================================================
# Extended node ID / label tests
# ===========================================================================


import uuid
from unittest.mock import MagicMock, call

import pytest

from graphknows.retrieval.stores.neo4j import (
    Neo4jStore,
    _GK_NS,
    _safe_rel_label,
    _write_relation_edges,
    chunk_node_id,
    entity_node_id,
)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestChunkNodeId:
    def test_returns_uuid_string(self):
        result = chunk_node_id("doc1", "hashABC")
        uuid.UUID(result)  # Raises if invalid UUID

    def test_deterministic(self):
        assert chunk_node_id("doc1", "hash") == chunk_node_id("doc1", "hash")

    def test_different_inputs_differ(self):
        assert chunk_node_id("doc1", "hash1") != chunk_node_id("doc1", "hash2")
        assert chunk_node_id("doc1", "hash") != chunk_node_id("doc2", "hash")

    def test_uses_gk_namespace(self):
        expected = str(uuid.uuid5(_GK_NS, "chunk:doc1:hash"))
        assert chunk_node_id("doc1", "hash") == expected


class TestEntityNodeId:
    def test_returns_uuid_string(self):
        result = entity_node_id("Alice")
        uuid.UUID(result)

    def test_case_insensitive(self):
        """alice vs Alice vs ALICE all produce same ID (normalised to upper)."""
        assert entity_node_id("alice") == entity_node_id("ALICE") == entity_node_id("Alice")

    def test_strips_whitespace(self):
        assert entity_node_id("  Alice  ") == entity_node_id("Alice")

    def test_different_names_differ(self):
        assert entity_node_id("Alice") != entity_node_id("Bob")


class TestSafeRelLabel:
    def test_lowercased_input_uppercased(self):
        assert _safe_rel_label("works at") == "WORKS_AT"

    def test_hyphens_to_underscores(self):
        assert _safe_rel_label("co-author") == "CO_AUTHOR"

    def test_already_valid(self):
        assert _safe_rel_label("REGULATES") == "REGULATES"

    def test_strips_non_word_chars(self):
        result = _safe_rel_label("foo!bar@baz")
        assert result == "FOOBARBAZ"

    def test_spaces_to_underscores(self):
        assert _safe_rel_label("part of") == "PART_OF"

    def test_empty_string(self):
        assert _safe_rel_label("") == ""

    def test_only_special_chars(self):
        assert _safe_rel_label("!!!") == ""


class TestWriteRelationEdges:
    def _make_mock_session(self):
        s = MagicMock()
        s.run = MagicMock(return_value=MagicMock())
        return s

    def test_writes_valid_edges(self):
        s = self._make_mock_session()
        entity_by_name = {
            "Alice": {"type": "PERSON"},
            "Bob": {"type": "PERSON"},
        }
        count = _write_relation_edges(s, [("Alice", "KNOWS", "Bob")], entity_by_name)
        assert count == 1
        s.run.assert_called_once()
        cypher = s.run.call_args[0][0]
        assert "KNOWS" in cypher
        assert "MERGE" in cypher

    def test_skips_unknown_endpoints(self):
        s = self._make_mock_session()
        entity_by_name = {"Alice": {}}
        count = _write_relation_edges(s, [("Alice", "KNOWS", "Unknown")], entity_by_name)
        assert count == 0
        s.run.assert_not_called()

    def test_skips_empty_relation_label(self):
        s = self._make_mock_session()
        entity_by_name = {"Alice": {}, "Bob": {}}
        count = _write_relation_edges(s, [("Alice", "!!!", "Bob")], entity_by_name)
        assert count == 0

    def test_multiple_edges(self):
        s = self._make_mock_session()
        entity_by_name = {"A": {}, "B": {}, "C": {}}
        count = _write_relation_edges(
            s,
            [("A", "KNOWS", "B"), ("B", "RELATES_TO", "C"), ("A", "RELATES_TO", "C")],
            entity_by_name,
        )
        assert count == 3
        assert s.run.call_count == 3

    def test_relation_label_normalized(self):
        s = self._make_mock_session()
        entity_by_name = {"A": {}, "B": {}}
        _write_relation_edges(s, [("A", "works for", "B")], entity_by_name)
        cypher = s.run.call_args[0][0]
        assert "WORKS_FOR" in cypher


# ---------------------------------------------------------------------------
# Neo4jStore class methods (mocked driver)
# ---------------------------------------------------------------------------


def _make_store():
    """Build a Neo4jStore with mocked neo4j driver — no DB connection needed."""
    store = Neo4jStore.__new__(Neo4jStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = iter([])
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    store._db = "neo4j"
    return store, mock_driver, mock_session


class TestNeo4jStoreLifecycle:
    def test_close(self):
        store, mock_driver, _ = _make_store()
        store.close()
        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_indexes_ignores_exceptions(self):
        store, _, mock_session = _make_store()
        mock_session.run.side_effect = Exception("already exists")
        # Should not raise
        await store.ensure_indexes()


class TestNeo4jStoreUpsertChunks:
    @pytest.mark.asyncio
    async def test_empty_chunks_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_chunks([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_chunks_cypher(self):
        store, _, mock_session = _make_store()
        await store.upsert_chunks([{"id": "c1", "doc_id": "d1", "text": "hello world"}])
        assert mock_session.run.called
        cypher = mock_session.run.call_args[0][0]
        assert ":CHUNK" in cypher
        assert "MERGE" in cypher

    @pytest.mark.asyncio
    async def test_upsert_chunks_truncates_text(self):
        store, _, mock_session = _make_store()
        long_text = "x" * 4000
        await store.upsert_chunks([{"id": "c1", "doc_id": "d", "text": long_text}])
        kwargs = mock_session.run.call_args[1]
        assert len(kwargs["text"]) <= 3000


class TestNeo4jStoreUpsertEntitiesAndMentions:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_entities_and_mentions([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_entity_and_mentions(self):
        store, _, mock_session = _make_store()
        chunks = [
            {
                "id": "c1",
                "entities": [{"text": "Alice", "label": "PERSON"}],
            }
        ]
        await store.upsert_entities_and_mentions(chunks)
        assert mock_session.run.call_count >= 2  # entity MERGE + MENTIONS MERGE
        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        assert any(":ENTITY" in cy for cy in all_cyphers)
        assert any("MENTIONS" in cy for cy in all_cyphers)

    @pytest.mark.asyncio
    async def test_skips_short_entity_names(self):
        """Entity names shorter than 2 chars should be skipped."""
        store, _, mock_session = _make_store()
        chunks = [{"id": "c1", "entities": [{"text": "a", "label": "X"}]}]
        await store.upsert_entities_and_mentions(chunks)
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_entity_label_fallback_chain(self):
        """label fallback: label → ontology_label → coarse_bucket → 'Entity'."""
        store, _, mock_session = _make_store()
        # All label fields absent → should default to 'Entity'
        chunks = [{"id": "c1", "entities": [{"text": "Bitcoin"}]}]
        await store.upsert_entities_and_mentions(chunks)
        entity_call = mock_session.run.call_args_list[0]
        assert entity_call[1]["label"] == "Entity"


class TestNeo4jStoreUpsertTopics:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_topics([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_topic_node_and_edges(self):
        store, _, mock_session = _make_store()
        await store.upsert_topics(
            [
                {
                    "id": "t1",
                    "label": "AI Safety",
                    "summary": "...",
                    "community_id": 0,
                    "modularity": 0.5,
                    "silhouette": 0.3,
                    "members": ["Alice", "Bob"],
                }
            ]
        )
        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        assert any(":TOPIC" in cy for cy in all_cyphers)
        assert any("IN_COMMUNITY" in cy for cy in all_cyphers)


class TestNeo4jStoreUpsertCoOccurs:
    @pytest.mark.asyncio
    async def test_empty_is_noop(self):
        store, _, mock_session = _make_store()
        await store.upsert_co_occurs([])
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_co_occurs_edge(self):
        store, _, mock_session = _make_store()
        await store.upsert_co_occurs([("Alice", "Bob", 3)])
        cypher = mock_session.run.call_args[0][0]
        assert "CO_OCCURS" in cypher
        assert mock_session.run.call_args[1]["w"] == 3


class TestNeo4jStoreGetChunksForEntity:
    @pytest.mark.asyncio
    async def test_returns_empty_if_no_results(self):
        store, _, mock_session = _make_store()
        mock_session.run.return_value = iter([])
        results = await store.get_chunks_for_entity("Alice")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_parsed_chunk_dicts(self):
        store, _, mock_session = _make_store()
        # _arun calls list(s.run(...)); each row is converted via dict(r)
        # So the mock row must support dict() conversion (i.e., items() or keys())
        fake_row = {"chunk_id": "c1", "doc_id": "d1", "text": "hello"}
        mock_session.run.return_value = iter([fake_row])

        results = await store.get_chunks_for_entity("Alice", top_k=3)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        cypher = mock_session.run.call_args[0][0]
        assert "MENTIONS" in cypher or ":ENTITY" in cypher


class TestNeo4jStoreArun:
    @pytest.mark.asyncio
    async def test_arun_delegates_to_run(self):
        store, _, mock_session = _make_store()
        mock_session.run.return_value = iter([])
        result = await store._arun("RETURN 1 AS n")
        assert isinstance(result, list)
