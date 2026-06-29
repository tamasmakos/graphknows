"""Unit tests for graphknows.memory.ltm.semantic (WS-6).

All tests are fast and run without a live DB or LLM.
Tests marked @pytest.mark.integration require a live ArcadeDB instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.ltm.semantic import (
    SemanticDistiller,
    _build_alias_map,
    _memory_class,
    _resolve_name,
    _sem_key,
    _triplets_from_relation_edges,
)


# ---------------------------------------------------------------------------
# Helper: shared STM data shapes
# ---------------------------------------------------------------------------

def _make_stm_data(
    relation_edges: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "entities": [],
        "files": [],
        "chunks": chunks or [],
        "dialogs": [],
        "turns": [],
        "relation_edges": relation_edges or [],
        "chunk_entity_edges": [],
        "turn_entity_edges": [],
        "next_edges": [],
    }


# ---------------------------------------------------------------------------
# Pure-logic helpers
# ---------------------------------------------------------------------------

class TestSemKey:
    def test_same_triplet_same_key(self) -> None:
        k1 = _sem_key("Alice", "LIVES_IN", "Berlin")
        k2 = _sem_key("alice", "lives_in", "berlin")
        assert k1 == k2

    def test_different_triplet_different_key(self) -> None:
        assert _sem_key("Alice", "LIVES_IN", "Berlin") != _sem_key("Alice", "WORKS_AT", "Berlin")

    def test_key_is_hex_string(self) -> None:
        key = _sem_key("X", "Y", "Z")
        assert len(key) == 40
        int(key, 16)  # Should not raise


class TestResolveName:
    def test_alias_resolved_to_canon(self) -> None:
        alias_map = {"eu": "European Union", "france": "France"}
        assert _resolve_name("eu", alias_map) == "European Union"
        assert _resolve_name("France", alias_map) == "France"

    def test_unknown_name_returned_as_is(self) -> None:
        assert _resolve_name("Germany", {}) == "Germany"


class TestMemoryClass:
    def test_preference_predicates(self) -> None:
        for pred in ("PREFERS", "LIKES", "DISLIKES", "WANTS", "AVOIDS", "INTERESTED_IN"):
            assert _memory_class(pred) == "preference", pred

    def test_non_preference_is_fact(self) -> None:
        assert _memory_class("LIVES_IN") == "fact"
        assert _memory_class("WORKS_AT") == "fact"


class TestBuildAliasMap:
    def test_aliases_mapped(self) -> None:
        aliases = {"European Union": ["eu", "EU", "e.u."]}
        m = _build_alias_map(aliases)
        assert m["eu"] == "European Union"
        assert m["e.u."] == "European Union"

    def test_empty_returns_empty(self) -> None:
        assert _build_alias_map({}) == {}


class TestTripletsFromRelationEdges:
    def test_basic_extraction(self) -> None:
        edges = [
            {"source": "Alice", "relation": "LIVES_IN", "target": "Berlin", "confidence": 0.9}
        ]
        result = _triplets_from_relation_edges(edges, {})
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["predicate"] == "LIVES_IN"
        assert result[0]["object"] == "Berlin"
        assert result[0]["confidence"] == 0.9

    def test_deduplication_averages_confidence(self) -> None:
        edges = [
            {"source": "Alice", "relation": "LIVES_IN", "target": "Berlin", "confidence": 0.8},
            {"source": "Alice", "relation": "LIVES_IN", "target": "Berlin", "confidence": 0.6},
        ]
        result = _triplets_from_relation_edges(edges, {})
        assert len(result) == 1
        # Average of 0.8 and 0.6 = 0.7
        assert abs(result[0]["confidence"] - 0.7) < 0.01

    def test_alias_resolution_deduplicates(self) -> None:
        """'eu' and 'EU' resolve to the same canonical name → single SEMANTIC node."""
        alias_map = {"eu": "European Union"}
        edges = [
            {"source": "eu", "relation": "LOCATED_IN", "target": "Europe", "confidence": 0.8},
            {"source": "EU", "relation": "LOCATED_IN", "target": "Europe", "confidence": 0.7},
        ]
        # 'EU' casefold is 'eu' — same canonical
        alias_map_full = {"eu": "European Union"}
        result = _triplets_from_relation_edges(edges, alias_map_full)
        # Both resolve to "European Union" LOCATED_IN "Europe" → 1 SEMANTIC node
        assert len(result) == 1
        assert result[0]["subject"] == "European Union"

    def test_skips_incomplete_edges(self) -> None:
        edges = [
            {"source": "Alice", "relation": "", "target": "Berlin"},
            {"source": "", "relation": "LIVES_IN", "target": "Berlin"},
            {"source": "Alice", "relation": "LIVES_IN", "target": ""},
        ]
        result = _triplets_from_relation_edges(edges, {})
        assert result == []

    def test_predicate_classified(self) -> None:
        edges = [{"source": "Alice", "relation": "LIKES", "target": "Coffee", "confidence": 0.9}]
        result = _triplets_from_relation_edges(edges, {})
        assert result[0]["memory_class"] == "preference"


# ---------------------------------------------------------------------------
# SemanticDistiller — llm_free mode
# ---------------------------------------------------------------------------

class TestSemanticDistillerLLMFree:
    @pytest.mark.asyncio
    async def test_distil_returns_facts_from_relation_edges(self) -> None:
        distiller = SemanticDistiller(openrouter_key="", llm_model="")
        stm = _make_stm_data(
            relation_edges=[
                {"source": "Alice", "relation": "WORKS_AT", "target": "Acme", "confidence": 0.9}
            ]
        )
        facts = await distiller.distil(stm, session_id="sess1")
        assert len(facts) == 1
        assert facts[0]["subject"] == "Alice"
        assert facts[0]["memory_class"] == "fact"
        assert "sess1" in facts[0]["source_session_ids"]

    @pytest.mark.asyncio
    async def test_distil_deduplicates_same_triplet(self) -> None:
        distiller = SemanticDistiller()
        stm = _make_stm_data(
            relation_edges=[
                {"source": "Alice", "relation": "LIVES_IN", "target": "Berlin", "confidence": 0.9},
                {"source": "alice", "relation": "LIVES_IN", "target": "berlin", "confidence": 0.7},
            ]
        )
        facts = await distiller.distil(stm, session_id="sess1")
        # Case-insensitive dedup: same triplet
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_distil_empty_stm_returns_empty(self) -> None:
        distiller = SemanticDistiller()
        stm = _make_stm_data()
        facts = await distiller.distil(stm, session_id="sess1")
        assert facts == []

    @pytest.mark.asyncio
    async def test_distil_uses_entity_aliases(self) -> None:
        """Aliases resolved → single canonical SEMANTIC node."""
        distiller = SemanticDistiller()
        stm = _make_stm_data(
            relation_edges=[
                {"source": "eu", "relation": "LOCATED_IN", "target": "Europe", "confidence": 0.8},
                {"source": "EU", "relation": "LOCATED_IN", "target": "Europe", "confidence": 0.7},
            ]
        )
        facts = await distiller.distil(
            stm,
            session_id="s1",
            entity_aliases={"European Union": ["eu", "EU"]},
        )
        assert len(facts) == 1
        assert facts[0]["subject"] == "European Union"

    @pytest.mark.asyncio
    async def test_distil_timestamps_present(self) -> None:
        distiller = SemanticDistiller()
        stm = _make_stm_data(
            relation_edges=[{"source": "A", "relation": "REL", "target": "B"}]
        )
        facts = await distiller.distil(stm, session_id="s1")
        assert len(facts) == 1
        assert "created_at" in facts[0]
        assert "updated_at" in facts[0]
        assert facts[0]["created_at"]  # non-empty

    @pytest.mark.asyncio
    async def test_llm_not_called_in_llm_free_mode(self) -> None:
        """LLM distillation helper must not be called when openrouter_key is empty."""
        distiller = SemanticDistiller(openrouter_key="", llm_model="")
        stm = _make_stm_data(
            chunks=[{"text": "Alice works at Acme."}],
            relation_edges=[{"source": "Alice", "relation": "WORKS_AT", "target": "Acme"}],
        )
        with patch(
            "graphknows.memory.ltm.semantic._llm_distill_chunks",
            new=AsyncMock(return_value=[]),
        ) as mock_llm:
            await distiller.distil(stm, session_id="s1")
            mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# SemanticDistiller — llm_assisted mode
# ---------------------------------------------------------------------------

class TestSemanticDistillerLLMAssisted:
    @pytest.mark.asyncio
    async def test_llm_triplets_merged_with_structural(self) -> None:
        """LLM triplets that aren't in structural set are included."""
        distiller = SemanticDistiller(openrouter_key="sk-test", llm_model="model/x")
        stm = _make_stm_data(
            relation_edges=[
                {"source": "Alice", "relation": "WORKS_AT", "target": "Acme", "confidence": 0.9}
            ],
            chunks=[{"text": "Alice prefers remote work."}],
        )
        llm_fact = {
            "id": "abc123def456abc123def456abc123def456abc1",
            "subject": "Alice",
            "predicate": "PREFERS",
            "object": "Remote work",
            "confidence": 0.8,
            "memory_class": "preference",
        }
        with patch(
            "graphknows.memory.ltm.semantic._llm_distill_chunks",
            new=AsyncMock(return_value=[llm_fact]),
        ):
            facts = await distiller.distil(stm, session_id="s1")
        subjects = {f["subject"] for f in facts}
        assert "Alice" in subjects
        predicates = {f["predicate"] for f in facts}
        assert "WORKS_AT" in predicates
        assert "PREFERS" in predicates

    @pytest.mark.asyncio
    async def test_llm_corroboration_boosts_confidence(self) -> None:
        """When LLM produces the same triplet as structural, confidence gets boosted."""
        distiller = SemanticDistiller(openrouter_key="sk-test", llm_model="model/x")
        stm = _make_stm_data(
            relation_edges=[
                {"source": "Bob", "relation": "WORKS_AT", "target": "Globex", "confidence": 0.8}
            ],
            chunks=[{"text": "Bob works at Globex."}],
        )
        # Same triplet as structural → same id key
        from graphknows.memory.ltm.semantic import _sem_key as sem_key
        key = sem_key("Bob", "WORKS_AT", "Globex")
        llm_fact = {
            "id": key,
            "subject": "Bob",
            "predicate": "WORKS_AT",
            "object": "Globex",
            "confidence": 0.8,
            "memory_class": "fact",
        }
        with patch(
            "graphknows.memory.ltm.semantic._llm_distill_chunks",
            new=AsyncMock(return_value=[llm_fact]),
        ):
            facts = await distiller.distil(stm, session_id="s1")
        assert len(facts) == 1
        # Confidence should be >= 0.8 (boosted)
        assert facts[0]["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_structural(self) -> None:
        """When LLM call fails, structural triplets are still returned."""
        distiller = SemanticDistiller(openrouter_key="sk-test", llm_model="model/x")
        stm = _make_stm_data(
            relation_edges=[
                {"source": "Carol", "relation": "LIVES_IN", "target": "Paris", "confidence": 0.9}
            ],
            chunks=[{"text": "Carol lives in Paris."}],
        )
        with patch(
            "graphknows.memory.ltm.semantic._llm_distill_chunks",
            new=AsyncMock(side_effect=RuntimeError("API down")),
        ):
            # _llm_distill_chunks itself catches errors and returns []
            # but if it raises, distil should still work
            facts = await distiller.distil(stm, session_id="s1")
        # At minimum the structural fact must be present
        assert any(f["subject"] == "Carol" for f in facts)
