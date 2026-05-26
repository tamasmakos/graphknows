"""
Tests for small, 0%-covered utility modules:
- graphknows.graphgen.data_types (Pydantic models)
- graphknows.graphgen.config.schema (GraphSchema, NodeSchema, EdgeSchema, get_default_schema)
- graphknows.forget (LTMForget — mocked neo4j driver)
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest


# ---------------------------------------------------------------------------
# data_types.py
# ---------------------------------------------------------------------------

from graphknows.graphgen.data_types import ChunkExtractionTask, SegmentData


class TestChunkExtractionTask:
    def test_defaults(self):
        task = ChunkExtractionTask(chunk_id="c1", chunk_text="hello")
        assert task.chunk_id == "c1"
        assert task.chunk_text == "hello"
        assert task.entities == []
        assert task.abstract_concepts == []
        assert task.keywords == []

    def test_with_values(self):
        task = ChunkExtractionTask(
            chunk_id="c2",
            chunk_text="Python is great",
            entities=["Python"],
            keywords=["programming"],
        )
        assert "Python" in task.entities
        assert "programming" in task.keywords

    def test_json_roundtrip(self):
        task = ChunkExtractionTask(chunk_id="x", chunk_text="y", entities=["A"])
        data = task.model_dump()
        restored = ChunkExtractionTask(**data)
        assert restored.chunk_id == "x"
        assert restored.entities == ["A"]


class TestSegmentData:
    def test_defaults(self):
        seg = SegmentData(segment_id="s1", content="text", line_number=1)
        assert seg.sentiment == 0.0
        assert seg.date is None

    def test_with_values(self):
        seg = SegmentData(segment_id="s1", content="abc", line_number=42, sentiment=0.8)
        assert seg.sentiment == 0.8
        assert seg.line_number == 42


# ---------------------------------------------------------------------------
# config/schema.py
# ---------------------------------------------------------------------------

from graphknows.graphgen.config.schema import (
    NodeSchema, EdgeSchema, GraphSchema, get_default_schema
)


class TestNodeSchema:
    def test_defaults(self):
        n = NodeSchema(label="Person")
        assert n.label == "Person"
        assert n.attributes == []
        assert n.source_type == "segment"

    def test_with_attrs(self):
        n = NodeSchema(label="Org", attributes=["name", "founded"], source_type="document")
        assert "name" in n.attributes
        assert n.source_type == "document"


class TestEdgeSchema:
    def test_construction(self):
        e = EdgeSchema(source_label="Person", target_label="Org", relation_type="WORKS_AT")
        assert e.relation_type == "WORKS_AT"
        assert not e.is_hierarchical

    def test_hierarchical(self):
        e = EdgeSchema(source_label="A", target_label="B", relation_type="PART_OF", is_hierarchical=True)
        assert e.is_hierarchical


class TestGraphSchema:
    def test_construction(self):
        gs = GraphSchema(
            nodes={"Person": NodeSchema(label="Person")},
            edges=[EdgeSchema(source_label="Person", target_label="Org", relation_type="KNOWS")],
        )
        assert "Person" in gs.nodes
        assert len(gs.edges) == 1


class TestGetDefaultSchema:
    def test_returns_graph_schema(self):
        schema = get_default_schema()
        assert isinstance(schema, GraphSchema)

    def test_has_nodes(self):
        schema = get_default_schema()
        assert len(schema.nodes) > 0

    def test_has_edges(self):
        schema = get_default_schema()
        assert len(schema.edges) > 0


# ---------------------------------------------------------------------------
# forget.py
# ---------------------------------------------------------------------------

from graphknows.forget import LTMForget


def _make_forget():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    return LTMForget(neo4j_driver=mock_driver, pg_dsn="postgresql://fake/fake"), mock_session


class TestLTMForget:
    def test_prune_disconnected_returns_count(self):
        forget, mock_session = _make_forget()
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, k: 5 if k == "deleted" else 0
        mock_session.run.return_value.single.return_value = mock_result
        result = forget.prune_disconnected()
        assert "deleted_entities" in result
        assert result["deleted_entities"] == 5

    def test_prune_disconnected_cypher(self):
        forget, mock_session = _make_forget()
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, k: 0
        mock_session.run.return_value.single.return_value = mock_result
        forget.prune_disconnected()
        cypher = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" in cypher

    def test_prune_disconnected_returns_zero_when_nothing_deleted(self):
        forget, mock_session = _make_forget()
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, k: 0
        mock_session.run.return_value.single.return_value = mock_result
        result = forget.prune_disconnected()
        assert result["deleted_entities"] == 0
