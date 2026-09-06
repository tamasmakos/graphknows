"""MemoryScope: structured agent/user/session identity for memory operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

from graphknows.exceptions import ConfigurationError


class MemoryScope(BaseModel, frozen=True):
    """Identity triple for scoping memory operations.

    At least one of user_id, agent_id, or run_id must be set.
    For multi-user deployments, always set user_id to prevent cross-user collisions.

    Attributes:
        user_id: The human participant identifier.
        agent_id: The agent or model identifier.
        run_id: A specific conversation run or session identifier.
    """

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    def __init__(self, **data: Any) -> None:
        # pydantic wraps the "at least one" check below into a ValidationError;
        # this is the public constructor path, so callers never have to catch
        # pydantic directly — only GraphKnowsError.
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise ConfigurationError(str(exc)) from exc

    @model_validator(mode="after")
    def _at_least_one(self) -> MemoryScope:
        if not any([self.user_id, self.agent_id, self.run_id]):
            raise ValueError("MemoryScope requires at least one of: user_id, agent_id, run_id")
        return self

    def to_session_id(self) -> str:
        """Produce a deterministic session_id string for storage backends.

        Single run_id with no other components returns the run_id as-is, preserving
        backward compatibility with callers that pass session_id directly.
        Multi-component scopes use a prefixed compound key to prevent collisions.
        """
        set_fields = [f for f in (self.user_id, self.agent_id, self.run_id) if f]
        if len(set_fields) == 1 and self.run_id and not (self.user_id or self.agent_id):
            return self.run_id
        parts = []
        if self.user_id:
            parts.append(f"u:{self.user_id}")
        if self.agent_id:
            parts.append(f"a:{self.agent_id}")
        if self.run_id:
            parts.append(f"r:{self.run_id}")
        return "|".join(parts)

    def to_dialog_id(self) -> str:
        """Derive a stable dialog_id for conversation threads."""
        return f"dialog-{self.to_session_id()}"
