"""Tests for the ADR-002 investigative state types and state construction."""

from __future__ import annotations

import uuid

from services.agent.graph.state import (
    Chunk,
    Entity,
    InvestigativeState,
    Relation,
    SubQuestion,
    WSEvent,
    build_initial_investigative_state,
)


class TestBuildInitialInvestigativeState:
    """Tests for build_initial_investigative_state."""

    def test_returns_correct_type(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="test topic")
        assert isinstance(state, dict)

    def test_sets_session_id(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="t")
        assert state["session_id"] == "s1"

    def test_sets_topic(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="EU AI Act")
        assert state["topic"] == "EU AI Act"

    def test_default_collections_are_empty(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="t")
        assert state["plan"] == []
        assert state["queries"] == []
        assert state["raw_docs"] == []
        assert state["chunks"] == []
        assert state["entities"] == []
        assert state["relations"] == []
        assert state["ws_events"] == []
        assert state["retrieval_hits"] == []

    def test_default_scalars(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="t")
        assert state["iteration_count"] == 0
        assert state["dup_ratio"] == 0.0
        assert state["briefing"] is None


class TestSubQuestionDataclass:
    """Tests for the SubQuestion dataclass."""

    def test_construction(self) -> None:
        sq = SubQuestion(id="sq-1", text="What is the EU AI Act?")
        assert sq.id == "sq-1"
        assert sq.text == "What is the EU AI Act?"
        assert sq.source_priorities == ["eprs_briefings", "ep_debates", "web"]

    def test_custom_source_priorities(self) -> None:
        sq = SubQuestion(id="sq-2", text="Test", source_priorities=["web"])
        assert sq.source_priorities == ["web"]


class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_construction(self) -> None:
        chunk = Chunk(
            id="chunk-1",
            doc_id="doc-1",
            text="Some passage text.",
        )
        assert chunk.id == "chunk-1"
        assert chunk.doc_id == "doc-1"
        assert chunk.text == "Some passage text."
        assert chunk.embedding is None
        assert chunk.metadata == {}

    def test_with_embedding_and_metadata(self) -> None:
        chunk = Chunk(
            id="chunk-2",
            doc_id="doc-2",
            text="Embedded text.",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata == {"source": "test"}


class TestEntityDataclass:
    """Tests for the Entity dataclass."""

    def test_construction(self) -> None:
        entity = Entity(
            id="ent-1",
            name="European Commission",
            label="ORG",
            confidence=0.95,
            doc_ids=["doc-1"],
        )
        assert entity.id == "ent-1"
        assert entity.name == "European Commission"
        assert entity.label == "ORG"
        assert entity.confidence == 0.95
        assert entity.doc_ids == ["doc-1"]

    def test_default_doc_ids(self) -> None:
        entity = Entity(id="ent-2", name="Test", label="PERSON", confidence=0.5)
        assert entity.doc_ids == []


class TestRelationDataclass:
    """Tests for the Relation dataclass."""

    def test_construction(self) -> None:
        rel = Relation(
            source="ent-1",
            target="ent-2",
            type="REGULATES",
            confidence=0.85,
        )
        assert rel.source == "ent-1"
        assert rel.target == "ent-2"
        assert rel.type == "REGULATES"
        assert rel.confidence == 0.85


class TestStateDictMergePattern:
    """Tests that InvestigativeState supports LangGraph-style dict merge updates."""

    def test_dict_merge(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="test")
        assert state["plan"] == []

        sq = SubQuestion(id="sq-1", text="Test question")
        update = {"plan": [sq], "iteration_count": 1}
        merged = {**state, **update}

        assert isinstance(merged, dict)
        assert merged["session_id"] == "s1"
        assert len(merged["plan"]) == 1
        assert merged["plan"][0].text == "Test question"
        assert merged["iteration_count"] == 1

    def test_append_ws_events(self) -> None:
        state = build_initial_investigative_state(session_id="s1", topic="test")
        event: WSEvent = {
            "session_id": "s1",
            "event_type": "plan_ready",
            "payload": {},
            "ts": 12345.0,
        }
        update = {"ws_events": [*state["ws_events"], event]}
        merged: InvestigativeState = {**state, **update}
        assert len(merged["ws_events"]) == 1
        assert merged["ws_events"][0]["event_type"] == "plan_ready"
