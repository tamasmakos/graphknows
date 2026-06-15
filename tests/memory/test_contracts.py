"""Contract tests for canonical memory DTOs."""

from __future__ import annotations

from graphknows.memory.adapter_types import MemoryClass
from graphknows.memory.models.contracts import (
    CommunityAssignment,
    MemoryRecord,
    RetrievalHit,
    SessionSnapshot,
)
from graphknows.memory.models.promotion import PromotionRequest


def test_session_snapshot_round_trips_store_shape() -> None:
    data = {
        "entities": [{"name": "EU"}],
        "files": [{"id": "f1"}],
        "chunks": [{"id": "c1"}],
        "dialogs": [],
        "turns": [],
        "chunk_entity_edges": [],
        "turn_entity_edges": [],
        "next_edges": [],
        "relation_edges": [],
        "_merge_count": 99,
    }

    snapshot = SessionSnapshot.from_store("s1", data)

    assert snapshot.session_id == "s1"
    assert snapshot.entities == [{"name": "EU"}]
    assert "_merge_count" not in snapshot.to_store_dict()


def test_promotion_request_defaults_to_topic_off_for_llm_free() -> None:
    policy = PromotionRequest(session_id="s1").resolve_policy("gliner2")

    assert policy.topic_mode == "off"
    assert policy.run_topics is False


def test_promotion_request_allows_llm_assisted_topic_enrichment() -> None:
    policy = PromotionRequest(session_id="s1").resolve_policy("llm_assisted")

    assert policy.topic_mode == "llm_assisted"
    assert policy.run_topics is True


def test_memory_record_carries_community_metadata() -> None:
    record = MemoryRecord(
        id="c1",
        session_id="s1",
        text="hello",
        memory_class=MemoryClass.fact,
        community_id=4,
        community_weight=2.5,
    )

    assert record.kind == "chunk"
    assert record.community_id == 4
    assert record.community_weight == 2.5


def test_retrieval_hit_and_community_assignment_contracts() -> None:
    hit = RetrievalHit(
        chunk_id="c1",
        text="policy",
        source="semantic",
        rank=0,
        metadata={"community_id": 1, "community_weight": 2.0},
    )
    assignment = CommunityAssignment(
        community_id=1,
        entity_names=["EU"],
        chunk_ids=["c1"],
        weight=2.0,
        algorithm="leiden",
    )

    assert hit.metadata["community_id"] == assignment.community_id
    assert assignment.chunk_ids == ["c1"]

