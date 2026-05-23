"""Shared fixture factory for synthetic host adapter conformance tests."""
from __future__ import annotations

from graphknows.memory.adapter_types import (
    ArtifactRecord,
    MemoryClass,
    ParticipantRecord,
    PromoteOptions,
    RecallRequest,
    ToolCallRecord,
    TurnRecord,
)

SESSION_ID = "test-session-001"
DIALOG_ID = "test-dialog-001"

PARTICIPANTS = [
    ParticipantRecord(participant_id="user-1", role="user", display_name="Alice"),
    ParticipantRecord(participant_id="agent-1", role="assistant", display_name="Agent"),
]

TURNS = [
    TurnRecord(
        turn_id="turn-0",
        session_id=SESSION_ID,
        role="user",
        content="What are the EU AI Act compliance requirements?",
        tool_calls=[],
        memory_class=MemoryClass.raw_evidence,
    ),
    TurnRecord(
        turn_id="turn-1",
        session_id=SESSION_ID,
        role="assistant",
        content="The EU AI Act requires risk classification...",
        tool_calls=[
            ToolCallRecord(
                tool_id="call-0",
                tool_name="web_search",
                input_summary="EU AI Act compliance 2025",
                output_summary="Found 10 results about risk categories",
                exit_code=None,
                error=None,
                memory_class=MemoryClass.raw_evidence,
            )
        ],
        memory_class=MemoryClass.raw_evidence,
    ),
]

ARTIFACTS = [
    ArtifactRecord(
        artifact_id="file-001",
        path="/root/graphknows/graphknows/memory/adapter.py",
        content="# adapter module source...\n",
        mime_type="text/x-python",
        memory_class=MemoryClass.raw_evidence,
        metadata={"lang": "python"},
    ),
]

RECALL_REQUEST = RecallRequest(
    query="EU AI Act compliance",
    session_id=SESSION_ID,
    top_k=5,
    include_raw=False,
)

PROMOTE_OPTIONS = PromoteOptions(
    session_id=SESSION_ID,
    run_gds=False,
    run_topics=False,
)
