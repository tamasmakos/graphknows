"""What a tool call evidences, read deterministically — no model, no decoder."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from graphknows.ingestion.extraction.protocol import PackGuidance, _byte_span, _entity_id
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind

# Tools that write. Everything else a tool call names a path for only read it.
_WRITING_TOOLS = frozenset({"Edit", "MultiEdit", "NotebookEdit", "Write"})
_PATH_FIELDS = ("file_path", "notebook_path", "path")


def _call(text: str) -> tuple[str, dict[str, Any]]:
    """The tool name and input of a ``name({...})`` segment; ``("", {})`` if it is not one."""
    name, _, rest = text.partition("(")
    if not name or not rest.endswith(")"):
        return "", {}
    try:
        arguments = json.loads(rest[:-1])
    except ValueError:
        return "", {}
    return (name, arguments) if isinstance(arguments, dict) else ("", {})


def _paths(arguments: dict[str, Any]) -> list[str]:
    """The file paths a tool call addressed, in field order."""
    return [
        str(arguments[field]) for field in _PATH_FIELDS if isinstance(arguments.get(field), str)
    ]


class AgentExtractor:
    """Turns the agent's own tool calls into ``touched`` and ``changed`` facts.

    Only ``tool_call`` segments carry evidence here: a tool result is a citable
    source, not an assertion, so the transcript parser never makes one a
    segment (FR-028). Decisions come from the pack's extraction guidance
    instead, off the user's and the assistant's prose.
    """

    name = "agent"
    version = "1"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def extract(self, segment: Segment, pack: PackGuidance) -> tuple[list[Mention], list[Fact]]:
        """The tool and file mentions of *segment*, and the fact that binds them."""
        if segment.kind is not SegmentKind.tool_call:
            return [], []
        tool, arguments = _call(segment.text)
        subject = self._mention(segment, tool, "tool")
        if subject is None:
            return [], []
        mentions: list[Mention] = [subject]
        facts: list[Fact] = []
        for path in _paths(arguments):
            object_ = self._mention(segment, path, "file")
            if object_ is None:
                continue
            mentions.append(object_)
            facts.append(self._fact(segment, subject, object_))
        return mentions, facts

    def _mention(self, segment: Segment, name: str, label: str) -> Mention | None:
        """*name* as a mention of *segment*, or ``None`` when the text cannot cite it.

        A path is cited at its JSON-escaped offsets: that is how the parser
        wrote the call's input into the segment's text.
        """
        surface = name if label == "tool" else json.dumps(name)[1:-1]
        span = _byte_span(segment.text, surface) if surface else None
        if span is None:
            return None
        return Mention(
            segment_id=segment.id,
            entity_id=_entity_id(name),
            surface=surface,
            span=span,
            label=label,
            extractor=self.name,
        )

    def _fact(self, segment: Segment, subject: Mention, object_: Mention) -> Fact:
        """The assertion that the tool acted on the file, cited by *segment*."""
        predicate = "agent:changed" if subject.surface in _WRITING_TOOLS else "agent:touched"
        key = f"{segment.id}|{subject.entity_id}|{predicate}|{object_.entity_id}"
        return Fact(
            id=sha256(key.encode()).hexdigest()[:16],
            subject=subject.entity_id,
            predicate=predicate,
            object=object_.entity_id,
            extractor=self.name,
            extractor_version=self.version,
        )


__all__ = ["AgentExtractor"]
