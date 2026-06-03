"""Tests for graphknows.memory.adapter_types."""

from __future__ import annotations

import pytest

from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    MemoryClass,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)


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


class TestArtifactRecord:
    """Tests for ArtifactRecord construction and invariants."""

    def test_construct_with_required_fields(self) -> None:
        """ArtifactRecord can be constructed with only required fields."""
        record = ArtifactRecord(
            artifact_id="a1",
            path="/tmp/file.py",
            content="print('hello')",
            mime_type="text/x-python",
        )

        assert record.artifact_id == "a1"
        assert record.path == "/tmp/file.py"
        assert record.content == "print('hello')"
        assert record.mime_type == "text/x-python"

    def test_defaults_are_applied(self) -> None:
        """Default values are set when optional fields are omitted."""
        record = ArtifactRecord(
            artifact_id="a1",
            path="/tmp/file.py",
            content="print('hello')",
            mime_type="text/x-python",
        )

        assert record.memory_class == MemoryClass.raw_evidence
        assert record.metadata == {}

    def test_memory_class_is_enum_value(self) -> None:
        """The memory_class field must be a MemoryClass enum value."""
        record = ArtifactRecord(
            artifact_id="a1",
            path="/tmp/file.py",
            content="print('hello')",
            mime_type="text/x-python",
            memory_class=MemoryClass.summary,
        )

        assert isinstance(record.memory_class, MemoryClass)
        assert record.memory_class == MemoryClass.summary

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen dataclass must raise when mutation is attempted."""
        record = ArtifactRecord(
            artifact_id="a1",
            path="/tmp/file.py",
            content="print('hello')",
            mime_type="text/x-python",
        )

        with pytest.raises(AttributeError, match="cannot assign to field"):
            record.artifact_id = "a2"


class TestTurnRecord:
    """Tests for TurnRecord construction and invariants."""

    def test_construct_with_required_fields(self) -> None:
        """TurnRecord can be constructed with only required fields."""
        record = TurnRecord(
            turn_id="t1",
            session_id="s1",
            role="assistant",
            content="Hello!",
        )

        assert record.turn_id == "t1"
        assert record.session_id == "s1"
        assert record.role == "assistant"
        assert record.content == "Hello!"

    def test_defaults_are_applied(self) -> None:
        """Default values are set when optional fields are omitted."""
        record = TurnRecord(
            turn_id="t1",
            session_id="s1",
            role="assistant",
            content="Hello!",
        )

        assert record.memory_class == MemoryClass.raw_evidence
        assert record.metadata == {}

    def test_memory_class_is_enum_value(self) -> None:
        """The memory_class field must be a MemoryClass enum value."""
        record = TurnRecord(
            turn_id="t1",
            session_id="s1",
            role="assistant",
            content="Hello!",
            memory_class=MemoryClass.episode,
        )

        assert isinstance(record.memory_class, MemoryClass)
        assert record.memory_class == MemoryClass.episode

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen dataclass must raise when mutation is attempted."""
        record = TurnRecord(
            turn_id="t1",
            session_id="s1",
            role="assistant",
            content="Hello!",
        )

        with pytest.raises(AttributeError, match="cannot assign to field"):
            record.turn_id = "t2"


class TestRecallRequest:
    """Tests for RecallRequest construction and invariants."""

    def test_construct_with_required_fields(self) -> None:
        """RecallRequest can be constructed with only required fields."""
        request = RecallRequest(query="find something")

        assert request.query == "find something"

    def test_defaults_are_applied(self) -> None:
        """Default values are set when optional fields are omitted."""
        request = RecallRequest(query="find something")

        assert request.session_id is None
        assert request.top_k == 10
        assert request.prefer_classes == [
            MemoryClass.state,
            MemoryClass.episode,
            MemoryClass.fact,
            MemoryClass.summary,
        ]
        assert request.include_raw is True

    def test_with_session_id(self) -> None:
        """RecallRequest accepts an explicit session_id."""
        request = RecallRequest(
            query="find something",
            session_id="session-42",
        )

        assert request.session_id == "session-42"

    def test_without_session_id(self) -> None:
        """RecallRequest defaults session_id to None."""
        request = RecallRequest(query="find something")

        assert request.session_id is None

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen dataclass must raise when mutation is attempted."""
        request = RecallRequest(query="find something")

        with pytest.raises(AttributeError, match="cannot assign to field"):
            request.query = "other"


class TestPromoteOptions:
    """Tests for PromoteOptions construction and invariants."""

    def test_construct_with_required_fields(self) -> None:
        """PromoteOptions can be constructed with only required fields."""
        options = PromoteOptions(session_id="s1")

        assert options.session_id == "s1"

    def test_defaults_are_applied(self) -> None:
        """Default values are set when optional fields are omitted."""
        options = PromoteOptions(session_id="s1")

        assert options.run_gds is True
        assert options.run_topics is True

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen dataclass must raise when mutation is attempted."""
        options = PromoteOptions(session_id="s1")

        with pytest.raises(AttributeError, match="cannot assign to field"):
            options.session_id = "s2"


class TestAdapterCapabilities:
    """Tests for AdapterCapabilities construction and invariants."""

    def test_construct_with_all_fields(self) -> None:
        """AdapterCapabilities can be constructed with all fields."""
        caps = AdapterCapabilities(
            supports_stm=True,
            supports_ltm=False,
            supports_dialog_graph=True,
            supports_artifact_ingest=False,
            supports_recall=True,
            supports_promote=False,
        )

        assert caps.supports_stm is True
        assert caps.supports_ltm is False
        assert caps.supports_dialog_graph is True
        assert caps.supports_artifact_ingest is False
        assert caps.supports_recall is True
        assert caps.supports_promote is False

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen dataclass must raise when mutation is attempted."""
        caps = AdapterCapabilities(
            supports_stm=True,
            supports_ltm=False,
            supports_dialog_graph=True,
            supports_artifact_ingest=False,
            supports_recall=True,
            supports_promote=False,
        )

        with pytest.raises(AttributeError, match="cannot assign to field"):
            caps.supports_stm = False
