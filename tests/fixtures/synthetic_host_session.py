"""Shared fixture factory for synthetic host adapter conformance tests."""

from __future__ import annotations

from graphknows.memory.adapter_types import (
    ArtifactRecord,
    MemoryClass,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)

SESSION_ID = "test-session-001"
DIALOG_ID = "test-dialog-001"

TURNS = [
    TurnRecord(
        turn_id="turn-0",
        session_id=SESSION_ID,
        role="user",
        content="What are the EU AI Act compliance requirements?",
        memory_class=MemoryClass.raw_evidence,
    ),
    TurnRecord(
        turn_id="turn-1",
        session_id=SESSION_ID,
        role="assistant",
        content="The EU AI Act requires risk classification...",
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
)
