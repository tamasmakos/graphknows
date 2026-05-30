"""Focused tests for topic recall ranking/filtering behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from graphknows.recall import recall_by_topic


def _make_driver(rows: list[dict]) -> MagicMock:
    mock_result = MagicMock()
    mock_result.data.return_value = rows
    mock_sess = MagicMock()
    mock_sess.run.return_value = mock_result
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_sess)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = ctx
    return mock_driver


def test_recall_by_topic_uses_in_topic_relationship() -> None:
    """The Cypher query must use IN_TOPIC, not COMPONENT_OF."""
    import inspect

    source = inspect.getsource(recall_by_topic)
    assert "IN_TOPIC" in source, "recall_by_topic must use IN_TOPIC relationship"
    assert "COMPONENT_OF" not in source, (
        "recall_by_topic must not use old COMPONENT_OF relationship"
    )


def test_recall_by_topic_matches_entity_concept_nodes() -> None:
    """Cypher must match ENTITY_CONCEPT nodes so IN_TOPIC join can resolve."""
    import inspect

    source = inspect.getsource(recall_by_topic)
    assert "(e:ENTITY OR e:ENTITY_CONCEPT)" in source, (
        "recall_by_topic Cypher must match both ENTITY and ENTITY_CONCEPT nodes"
    )


def test_recall_by_topic_entity_concept_returns_nonempty_entities() -> None:
    """When driver returns an ENTITY_CONCEPT name, entities must be non-empty and score > 0."""
    rows = [
        {
            "topic_id": 42,
            "summary": "Topic about memory architecture",
            "entities": ["memory"],
            "score": 3,
        }
    ]
    driver = _make_driver(rows)

    hits = recall_by_topic(driver, "memory architecture agents", top_k=5)

    assert len(hits) == 1
    assert hits[0].entity_names != [], "entities must be non-empty when ENTITY_CONCEPT matched"
    assert hits[0].score > 0, "score must be > 0 when overlap exists"


def test_recall_by_topic_returns_hits() -> None:
    """recall_by_topic returns RecallHit objects from driver results."""
    rows = [
        {"topic_id": 1, "summary": "Test topic", "entities": ["entity_a"], "score": 1},
    ]
    driver = _make_driver(rows)

    hits = recall_by_topic(driver, "entity_a test", top_k=5)

    assert len(hits) == 1
    assert hits[0].text == "Test topic"
    assert hits[0].entity_names == ["entity_a"]
    assert hits[0].score == 1.0


def test_recall_by_topic_filters_unrelated_zero_score_rows() -> None:
    """Unrelated topics with no entity or text overlap should be dropped."""
    rows = [
        {"topic_id": 1, "summary": "Monetary policy overview", "entities": [], "score": 0},
    ]
    driver = _make_driver(rows)

    hits = recall_by_topic(driver, "volcano geology", top_k=5)

    assert hits == []


def test_recall_by_topic_returns_summary_text_match_without_entity_overlap() -> None:
    """Summary/label text overlap alone should produce a relevant hit."""
    rows = [
        {
            "topic_id": 7,
            "summary": "Climate adaptation and emissions planning",
            "label": "environment policy",
            "entities": [],
            "score": 0,
        },
    ]
    driver = _make_driver(rows)

    hits = recall_by_topic(driver, "climate emissions", top_k=5)

    assert len(hits) == 1
    assert hits[0].text == "Climate adaptation and emissions planning"
    assert hits[0].entity_names == []
    assert hits[0].score > 0


def test_recall_by_topic_top_k_clamped_to_twenty() -> None:
    """Top-k should be clamped to the public API maximum."""
    driver = _make_driver([])

    recall_by_topic(driver, "climate policy topic", top_k=999)

    assert (
        driver.session.return_value.__enter__.return_value.run.call_args.kwargs["top_k"] == 20
    )


def test_recall_by_topic_empty_tokens_returns_empty() -> None:
    """When no tokens are extracted, return empty list without hitting driver."""
    driver = MagicMock()
    hits = recall_by_topic(driver, "a an", top_k=5)  # all tokens < 3 chars
    assert hits == []
    driver.session.assert_not_called()
