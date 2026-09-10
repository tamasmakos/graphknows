"""The agent pack: what an agent session, its prompts and its decisions are."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

from graphknows.models import ConceptRef, PredicateRef
from graphknows.models.fact import Fact, FactState

DECIDED_IN = "agent:decided_in"
DECISION_STATUS = "agent:decision_status"

_PROMPT_ADDENDUM = (
    "Label as `decision` any choice the user or the assistant settles on — an approach "
    "taken, a design fixed, an option rejected — naming it by the words that state the "
    "choice, not by the reasoning around it. A question, a plan still under discussion "
    "and a tool's output are not decisions."
)


@cache
def _data() -> Mapping[str, list[dict[str, Any]]]:
    """The pack's shipped symbol tables, read once."""
    raw = files("graphknows.packs").joinpath("data/agent.json").read_text(encoding="utf-8")
    data: Mapping[str, list[dict[str, Any]]] = json.loads(raw)
    return data


def _is_nameable(surface: str) -> bool:
    """Admit a surface as an entity name: anything the transcript actually wrote."""
    return bool(surface.strip())


@dataclass(frozen=True, slots=True)
class AgentPack:
    """The agent loop's own vocabulary, loaded beside any other pack (FR-004)."""

    name: str = "agent"

    def concepts(self) -> Iterable[ConceptRef]:
        """The five concepts an agent session is described with, iterated lazily."""
        for raw in _data()["concepts"]:
            yield ConceptRef(pack=self.name, **raw)

    def predicates(self) -> Iterable[PredicateRef]:
        """The relations the pack names, ``decision_status`` among them, lazily."""
        for raw in _data()["predicates"]:
            yield PredicateRef(pack=self.name, **raw)

    def entity_labels(self) -> tuple[str, ...]:
        """The label set offered to the decoder; it replaces the core's."""
        return ("session", "prompt", "decision", "file", "tool")

    def extractor(self) -> Any:
        """This pack reads its own domain: tool calls and stated decisions (FR-030)."""
        from graphknows.packs.agent.extractor import AgentExtractor

        return AgentExtractor()

    def prompt_addendum(self) -> str:
        """How to spot a decision in what the user and the assistant wrote."""
        return _PROMPT_ADDENDUM

    def hygiene(self) -> Callable[[str], bool]:
        """The pack's name hygiene: a transcript names its own files and tools."""
        return _is_nameable

    def thresholds(self) -> Mapping[str, float]:
        """A decision read out of prose is worth keeping only when the decoder is sure."""
        return {"mention": 0.5, "fact": 0.5}

    def veto(self, a: str, b: str) -> bool:
        """The pack forbids no merge: its entities are paths and tool names."""
        return False


def open_decisions(facts: Iterable[Fact]) -> list[Fact]:
    """The decisions among *facts* that nothing has since closed (FR-032).

    A decision is open while no fact on the functional ``decision_status``
    predicate supersedes it — that predicate is what settles a decision, so its
    absence is what leaves one open.
    """
    known = list(facts)
    closed = {
        fact.subject
        for fact in known
        if fact.predicate == DECISION_STATUS and fact.is_current and fact.state is FactState.ACTIVE
    }
    return [fact for fact in known if fact.predicate == DECIDED_IN and fact.subject not in closed]


__all__ = ["DECIDED_IN", "DECISION_STATUS", "AgentPack", "open_decisions"]
