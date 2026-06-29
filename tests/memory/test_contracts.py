"""Contract tests for canonical memory DTOs."""

from __future__ import annotations

from graphknows.memory.models.contracts import (
    CommunityAssignment,
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
    policy = PromotionRequest(session_id="s1").resolve_policy(extraction_uses_llm=False)

    assert policy.topic_mode == "off"
    assert policy.run_topics is False
    # Community detection is inert in llm_free (no topics, weight unconsumed) — skip it.
    assert policy.run_gds is False


def test_promotion_request_allows_llm_assisted_topic_enrichment() -> None:
    policy = PromotionRequest(session_id="s1").resolve_policy(extraction_uses_llm=True)

    assert policy.topic_mode == "llm_assisted"
    assert policy.run_topics is True
    # GDS runs in llm_assisted: community_id feeds topics + community-weighted fusion.
    assert policy.run_gds is True


def test_community_assignment_contract() -> None:
    assignment = CommunityAssignment(
        community_id=1,
        entity_names=["EU"],
        chunk_ids=["c1"],
        weight=2.0,
        algorithm="leiden",
    )

    assert assignment.community_id == 1
    assert assignment.chunk_ids == ["c1"]
