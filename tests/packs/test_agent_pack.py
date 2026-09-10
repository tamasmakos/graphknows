"""The agent pack: its symbols, its tool-call facts, and what leaves a decision open."""

from __future__ import annotations

import json

from graphknows.ingestion.extraction.protocol import Extractor
from graphknows.models.fact import Fact, FactState
from graphknows.models.segment import Segment, SegmentKind
from graphknows.packs import DomainPack, load_packs
from graphknows.packs.agent import AgentExtractor, AgentPack, open_decisions


def _tool_call(name: str, arguments: dict[str, object]) -> Segment:
    """A segment shaped like the transcript parser's rendering of a ``tool_use`` block."""
    return Segment(
        source_id="s1",
        text=f"{name}({json.dumps(arguments, sort_keys=True)})",
        kind=SegmentKind.tool_call,
        path="uuid/0",
    )


def _decision_fact(id_: str, subject: str, predicate: str) -> Fact:
    return Fact(id=id_, subject=subject, predicate=predicate, object="session")


def test_the_pack_satisfies_the_protocols() -> None:
    assert isinstance(AgentPack(), DomainPack)
    assert isinstance(AgentExtractor(), Extractor)


def test_the_pack_ships_its_five_concepts_and_five_predicates() -> None:
    pack = AgentPack()

    concepts = {concept.uri: concept for concept in pack.concepts()}
    predicates = {predicate.id: predicate for predicate in pack.predicates()}

    assert set(concepts) == {
        "agent:session",
        "agent:prompt",
        "agent:decision",
        "agent:file",
        "agent:tool",
    }
    assert set(predicates) == {
        "agent:asked_in",
        "agent:touched",
        "agent:changed",
        "agent:decided_in",
        "agent:decision_status",
    }
    assert all(symbol.pack == "agent" for symbol in (*concepts.values(), *predicates.values()))


def test_only_decision_status_is_functional() -> None:
    functional = {predicate.id for predicate in AgentPack().predicates() if predicate.functional}
    assert functional == {"agent:decision_status"}


def test_the_pack_loads_beside_an_unrelated_pack() -> None:
    loaded = load_packs([AgentPack()])
    assert loaded.concepts["agent:decision"].label == "decision"


def test_guidance_labels_decisions_from_prose() -> None:
    pack = AgentPack()
    assert "decision" in pack.entity_labels()
    assert "decision" in pack.prompt_addendum()


def test_a_write_tool_changes_the_file_it_named() -> None:
    segment = _tool_call("Edit", {"file_path": "src/app.py", "old_string": "a"})

    mentions, facts = AgentExtractor().extract(segment, AgentPack())

    assert [mention.label for mention in mentions] == ["tool", "file"]
    assert [fact.predicate for fact in facts] == ["agent:changed"]
    assert facts[0].subject == mentions[0].entity_id
    assert facts[0].object == mentions[1].entity_id


def test_a_reading_tool_only_touches_the_file() -> None:
    segment = _tool_call("Read", {"file_path": "src/app.py"})

    _, facts = AgentExtractor().extract(segment, AgentPack())

    assert [fact.predicate for fact in facts] == ["agent:touched"]


def test_every_mention_cites_bytes_of_its_segment() -> None:
    segment = _tool_call("Read", {"file_path": "src/app.py"})

    mentions, _ = AgentExtractor().extract(segment, AgentPack())

    encoded = segment.text.encode()
    assert all(
        encoded[start:end].decode() == mention.surface
        for mention in mentions
        for start, end in [mention.span]
    )


def test_a_tool_call_naming_no_path_evidences_no_fact() -> None:
    segment = _tool_call("Bash", {"command": "ls"})

    mentions, facts = AgentExtractor().extract(segment, AgentPack())

    assert [mention.label for mention in mentions] == ["tool"]
    assert facts == []


def test_prose_settling_nothing_evidences_nothing() -> None:
    segment = Segment(source_id="s1", text="Read(x)", kind=SegmentKind.turn, path="uuid/1")
    assert AgentExtractor().extract(segment, AgentPack()) == ([], [])


def _prose(text: str, role: str = "") -> Segment:
    return Segment(source_id="s1", text=text, kind=SegmentKind.prose, path="uuid/1", role=role)


def test_a_settled_choice_is_a_decision_of_its_session() -> None:
    segment = _prose("The agent decided to rebuild the namespace. It then did so.")

    mentions, facts = AgentExtractor().extract(segment, AgentPack())

    assert [mention.surface for mention in mentions] == ["rebuild the namespace", "s1"]
    assert [mention.label for mention in mentions] == ["decision", "session"]
    assert [fact.predicate for fact in facts] == ["agent:decided_in"]
    assert (facts[0].subject, facts[0].object) == (mentions[0].entity_id, mentions[1].entity_id)


def test_a_decision_is_cited_where_the_segment_states_it() -> None:
    segment = _prose("The agent decided to rebuild the namespace. It then did so.")

    mentions, _ = AgentExtractor().extract(segment, AgentPack())

    start, end = mentions[0].span
    assert segment.text.encode()[start:end].decode() == "rebuild the namespace"


def test_what_the_user_asked_was_asked_in_the_session() -> None:
    segment = _prose("Rebuild the namespace. Then run the gate.", role="user")

    mentions, facts = AgentExtractor().extract(segment, AgentPack())

    assert [mention.surface for mention in mentions] == ["Rebuild the namespace", "s1"]
    assert [fact.predicate for fact in facts] == ["agent:asked_in"]


def test_what_the_assistant_wrote_is_not_a_prompt() -> None:
    segment = _prose("Rebuild the namespace.", role="assistant")

    assert AgentExtractor().extract(segment, AgentPack()) == ([], [])


def test_a_decision_is_open_until_a_status_supersedes_it() -> None:
    open_one = _decision_fact("f1", "decision-a", "agent:decided_in")
    closed_one = _decision_fact("f2", "decision-b", "agent:decided_in")
    status = _decision_fact("f3", "decision-b", "agent:decision_status")

    assert open_decisions([open_one, closed_one, status]) == [open_one]


def test_a_forgotten_status_leaves_its_decision_open() -> None:
    decision = _decision_fact("f1", "decision-a", "agent:decided_in")
    status = _decision_fact("f2", "decision-a", "agent:decision_status").model_copy(
        update={"state": FactState.FORGOTTEN}
    )

    assert open_decisions([decision, status]) == [decision]
