"""Unit tests for neo4j.py module-level helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from graphknows.retrieval.stores.neo4j import (
    _safe_rel_label,
    _write_evidence_graph,
    _write_relation_edges,
)


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


class TestWriteRelationEdges:
    def test_dynamic_label_used(self) -> None:
        """Relation type must appear as the Cypher relationship type, not as a property."""
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
        """Triplets with endpoints not in entity_by_name are silently skipped."""
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
        """Lower-case labels must be uppercased in the query."""
        s = _mock_session()
        eb = {"EU": {}, "ECB": {}}
        triplets = [("EU", "funds", "ECB")]
        count = _write_relation_edges(s, triplets, eb)
        assert count == 1
        call_str = str(s.run.call_args_list[0])
        assert "FUNDS" in call_str
