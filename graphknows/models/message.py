"""Message normalization: the single entry point for all dialogue input."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from graphknows.bounds import MAX_TEXT_CHARS
from graphknows.exceptions import ConfigurationError


class Message(BaseModel, frozen=True):
    """A single conversation message.

    Attributes:
        role: Speaker role, as a free string: a source may name speakers no
            closed vocabulary has a member for.
        content: Text content.
        name: Optional speaker alias or tool name.
        tool_call_id: For role=tool replies, the originating tool call id.
        timestamp: When the turn was said, as a date/datetime string. This is
            the anchor relative dates resolve against: without it "next Friday"
            or "20 January" in a turn resolves against *ingestion* time, which
            silently dates a 2023 conversation to the year it was imported.
    """

    role: str
    content: str = Field(max_length=MAX_TEXT_CHARS)
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: str | None = None

    def __init__(self, **data: Any) -> None:
        # pydantic wraps the content-length bound (FR-030) into a
        # ValidationError; this is the public constructor path, so callers
        # never have to catch pydantic directly — only GraphKnowsError,
        # matching MemoryScope.__init__.
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise ConfigurationError(str(exc)) from exc


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
        return [Message(role="user", content=input)] if stripped else []
    if isinstance(input, dict):
        if "content" not in input:
            text = str(input).strip()
            return [Message(role="user", content=str(input))] if text else []
        msg = Message(**input)
        return [msg] if msg.content.strip() else []
    # list branch
    result: list[Message] = []
    for item in input:
        result.extend(normalize_messages(item))
    return result
