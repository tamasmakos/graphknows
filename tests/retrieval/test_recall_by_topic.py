"""Test recall_by_topic uses the correct IN_TOPIC relationship."""

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
    assert "COMPONENT_OF" not in source, "recall_by_topic must not use old COMPONENT_OF relationship"


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


def test_recall_by_topic_empty_tokens_returns_empty() -> None:
    """When no tokens are extracted, return empty list without hitting driver."""
    driver = MagicMock()
    hits = recall_by_topic(driver, "a an", top_k=5)  # all tokens < 3 chars
    assert hits == []
    driver.session.assert_not_called()
