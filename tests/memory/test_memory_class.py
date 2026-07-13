"""Tests for graphknows.models.memory_class."""

from __future__ import annotations

from graphknows.models.memory_class import MemoryClass


class TestMemoryClass:
    """Tests for the MemoryClass enum."""

    def test_string_values_match_expected_literals(self) -> None:
        """MemoryClass string values must match the specification."""
        assert MemoryClass.raw_evidence == "raw_evidence"
        assert MemoryClass.summary == "summary"
        assert MemoryClass.state == "state"
        assert MemoryClass.episode == "episode"
        assert MemoryClass.fact == "fact"
        assert MemoryClass.other == "other"
