"""Message normalization: the single entry point for all dialogue input."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class Role(StrEnum):
    """Supported conversation message roles."""

    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class Message(BaseModel, frozen=True):
    """A single conversation message.

    Attributes:
        role: Speaker role.
        content: Text content.
        name: Optional speaker alias or tool name.
        tool_call_id: For role=tool replies, the originating tool call id.
        timestamp: When the turn was said, as a date/datetime string. This is
            the anchor relative dates resolve against: without it "next Friday"
            or "20 January" in a turn resolves against *ingestion* time, which
            silently dates a 2023 conversation to the year it was imported.
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_role_aliases(cls, v: object) -> object:
        if isinstance(v, dict) and v.get("role") == "agent":
            v = dict(v, role="assistant")
        return v


MessageInput = str | dict[str, Any] | list[Any]


def normalize_messages(input: MessageInput) -> list[Message]:
    """Normalize any message input to a list of Messages.

    Accepts:
    - str  → [Message(role=user, content=str)]
    - dict → [Message(**dict)] or [Message(role=user, content=str(dict))]
    - list[str | dict | Message] → each item through the same rules

    Empty-content items are filtered out to prevent ghost graph nodes.
    """
    if isinstance(input, Message):
        return [input] if input.content.strip() else []
    if isinstance(input, str):
        stripped = input.strip()
        return [Message(role=Role.user, content=input)] if stripped else []
    if isinstance(input, dict):
        if "content" not in input:
            text = str(input).strip()
            return [Message(role=Role.user, content=str(input))] if text else []
        msg = Message(**input)
        return [msg] if msg.content.strip() else []
    # list branch
    result: list[Message] = []
    for item in input:
        result.extend(normalize_messages(item))
    return result
