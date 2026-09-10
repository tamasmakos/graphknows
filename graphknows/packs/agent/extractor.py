"""What a session evidences, read deterministically — no model, no decoder."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from graphknows.ingestion.extraction.protocol import PackGuidance, _byte_span, _entity_id
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind
from graphknows.packs.agent.pack import DECIDED_IN

# Tools that write. Everything else a tool call names a path for only read it.
_WRITING_TOOLS = frozenset({"Edit", "MultiEdit", "NotebookEdit", "Write"})
_PATH_FIELDS = ("file_path", "notebook_path", "path")

_ASKED_IN = "agent:asked_in"

# What was written is prose; what was called is not.
_STATED = frozenset({SegmentKind.prose, SegmentKind.turn})

# The words a transcript settles a choice with. What follows them, up to the
# end of the sentence, is the choice itself.
_DECIDED = re.compile(r"\bdecided\s+(?:to|that|on|against|in favour of)\s+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?\n]")


@dataclass(frozen=True, slots=True)
class _Claim:
    """One relation the transcript asserts: *subject* ``predicate`` *object*."""

    subject: Mention
    predicate: str
    object: Mention


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


def _sentence(text: str, start: int) -> str:
    """The rest of the sentence beginning at *start*, without its terminator."""
    end = _SENTENCE_END.search(text, start)
    return text[start : end.start() if end else len(text)].strip()


def _decision(text: str) -> str | None:
    """The choice *text* states, or ``None`` when it states none.

    Only a settled choice is a decision (FR-032): the pack's own guidance says
    so for the assisted path, and the same rule reads here as the words that
    settle it — "decided to", "decided against" — followed by the choice.
    """
    stated = _DECIDED.search(text)
    return _sentence(text, stated.end()) or None if stated else None


class AgentExtractor:
    """Turns the agent's own session into the facts its transcript evidences.

    Two kinds of segment carry evidence. A ``tool_call`` says which files the
    agent ``touched`` and ``changed``; prose says which prompts were asked and
    which decisions were settled, both ``in`` the session the segment belongs
    to. A tool result is a citable source, not an assertion, so the transcript
    parser never makes one a segment (FR-028).
    """

    name = "agent"
    version = "1"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def extract(self, segment: Segment, pack: PackGuidance) -> tuple[list[Mention], list[Fact]]:
        """The mentions and facts *segment* evidences under the pack's hygiene."""
        if segment.kind is SegmentKind.tool_call:
            named, claims = self._tool_call(segment)
        elif segment.kind in _STATED:
            named, claims = [], self._stated(segment)
        else:
            return [], []
        admits = pack.hygiene()
        kept = [claim for claim in claims if admits(claim.subject.surface)]
        mentions = {
            mention.entity_id: mention
            for mention in named + [end for claim in kept for end in (claim.subject, claim.object)]
        }
        return list(mentions.values()), [self._fact(segment, claim) for claim in kept]

    def _tool_call(self, segment: Segment) -> tuple[list[Mention], list[_Claim]]:
        """The tool this segment called, and what it did to each file it addressed.

        The tool is mentioned whether or not it named a file: a call is
        evidence the tool ran, and only the relation to a file needs one.
        """
        tool, arguments = _call(segment.text)
        subject = self._mention(segment, tool, "tool")
        if subject is None:
            return [], []
        predicate = "agent:changed" if tool in _WRITING_TOOLS else "agent:touched"
        return [subject], [
            _Claim(subject, predicate, object_)
            for path in _paths(arguments)
            if (object_ := self._mention(segment, path, "file")) is not None
        ]

    def _stated(self, segment: Segment) -> list[_Claim]:
        """The prompt asked and the decision settled in this segment, if any.

        A prompt is what the user wrote, taken as its opening sentence: the
        request is the ask, and the reasoning after it is not a second one.
        """
        session = self._session(segment)
        asked = _sentence(segment.text, 0) if segment.role == "user" else ""
        stated = [
            (_ASKED_IN, asked, "prompt"),
            (DECIDED_IN, _decision(segment.text) or "", "decision"),
        ]
        return [
            _Claim(subject, predicate, session)
            for predicate, surface, label in stated
            if surface and (subject := self._mention(segment, surface, label)) is not None
        ]

    def _session(self, segment: Segment) -> Mention:
        """The session this segment was written in, cited at the segment itself.

        A transcript never spells its own session id, and the segment is the
        evidence for it either way (FR-008).
        """
        return Mention(
            segment_id=segment.id,
            entity_id=_entity_id(segment.source_id),
            surface=segment.source_id,
            span=(0, len(segment.text.encode())),
            label="session",
            extractor=self.name,
        )

    def _mention(self, segment: Segment, name: str, label: str) -> Mention | None:
        """*name* as a mention of *segment*, or ``None`` when the text cannot cite it.

        A path is cited at its JSON-escaped offsets: that is how the parser
        wrote the call's input into the segment's text.
        """
        surface = json.dumps(name)[1:-1] if label == "file" else name
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

    def _fact(self, segment: Segment, claim: _Claim) -> Fact:
        """*claim* as a fact anchored to the segment that evidences it."""
        key = f"{segment.id}|{claim.subject.entity_id}|{claim.predicate}|{claim.object.entity_id}"
        return Fact(
            id=sha256(key.encode()).hexdigest()[:16],
            subject=claim.subject.entity_id,
            predicate=claim.predicate,
            object=claim.object.entity_id,
            extractor=self.name,
            extractor_version=self.version,
        )


__all__ = ["AgentExtractor"]
