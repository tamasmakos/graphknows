"""Unit tests for EntityResolution.resolve()."""

from __future__ import annotations

import pytest

from graphknows.memory.ltm.entity_resolution import EntityResolution


def _resolve(entities):
    return EntityResolution().resolve(entities)


def test_empty_input_returns_empty():
    resolved, audit = _resolve([])
    assert resolved == []
    assert audit.merged_entities == 0


def test_single_entity_no_merge():
    resolved, audit = _resolve([{"name": "EU", "type": "ORG", "confidence": 0.9}])
    assert len(resolved) == 1
    assert resolved[0]["name"] == "EU"
    assert audit.merged_entities == 0


def test_case_insensitive_merge_keeps_highest_confidence():
    entities = [
        {"name": "EU", "type": "ORG", "confidence": 0.9},
        {"name": "eu", "type": "ORG", "confidence": 0.5},
    ]
    resolved, audit = _resolve(entities)
    assert len(resolved) == 1
    assert resolved[0]["name"] == "EU"
    assert resolved[0]["confidence"] == 0.9
    assert audit.merged_entities == 1


def test_aliases_recorded_on_canonical():
    entities = [
        {"name": "France", "type": "LOC", "confidence": 0.8},
        {"name": "france", "type": "LOC", "confidence": 0.6},
    ]
    resolved, audit = _resolve(entities)
    assert len(resolved) == 1
    assert "france" in resolved[0]["aliases"]
    assert "France" in audit.entity_aliases


def test_unique_entities_all_preserved():
    entities = [
        {"name": "Germany", "type": "LOC", "confidence": 0.9},
        {"name": "France", "type": "LOC", "confidence": 0.8},
        {"name": "Spain", "type": "LOC", "confidence": 0.7},
    ]
    resolved, audit = _resolve(entities)
    assert len(resolved) == 3
    assert audit.merged_entities == 0


def test_entities_without_confidence_treated_as_one():
    entities = [
        {"name": "Acme", "type": "ORG"},
        {"name": "acme", "type": "ORG"},
    ]
    resolved, audit = _resolve(entities)
    assert len(resolved) == 1
    assert audit.merged_entities == 1


def test_empty_name_skipped():
    entities = [
        {"name": "", "type": "ORG", "confidence": 0.9},
        {"name": "EU", "type": "ORG", "confidence": 0.8},
    ]
    resolved, audit = _resolve(entities)
    assert len(resolved) == 1
    assert resolved[0]["name"] == "EU"
