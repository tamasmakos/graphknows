"""Loading packs is all-or-nothing, and a contested key names both packs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import pytest

from graphknows.exceptions import PackConflictError
from graphknows.models import ConceptRef, PredicateRef
from graphknows.packs import DomainPack, load_packs


@dataclass(frozen=True)
class _StubPack:
    """A pack that supplies exactly the symbols a test hands it."""

    name: str
    _concepts: tuple[ConceptRef, ...] = ()
    _predicates: tuple[PredicateRef, ...] = ()
    seen: list[str] = field(default_factory=list)

    def concepts(self) -> Iterable[ConceptRef]:
        for concept in self._concepts:
            self.seen.append(concept.uri)
            yield concept

    def predicates(self) -> Iterable[PredicateRef]:
        return iter(self._predicates)

    def entity_labels(self) -> tuple[str, ...]:
        return ()

    def prompt_addendum(self) -> str:
        return ""

    def hygiene(self) -> Callable[[str], bool]:
        return lambda surface: True

    def thresholds(self) -> Mapping[str, float]:
        return {}

    def veto(self, a: str, b: str) -> bool:
        return False


def _concept(uri: str, pack: str) -> ConceptRef:
    return ConceptRef(uri=uri, label=uri, definition=f"The {uri} concept.", pack=pack)


def _predicate(id_: str, pack: str) -> PredicateRef:
    return PredicateRef(
        id=id_, label=id_, definition=f"The {id_} relation.", canonical=id_, pack=pack
    )


def test_stub_pack_satisfies_the_protocol() -> None:
    assert isinstance(_StubPack(name="code"), DomainPack)


def test_two_unrelated_packs_share_one_namespace() -> None:
    code = _StubPack(
        name="code",
        _concepts=(_concept("code:function", "code"),),
        _predicates=(_predicate("code:calls", "code"),),
    )
    agent = _StubPack(
        name="agent",
        _concepts=(_concept("agent:decision", "agent"),),
        _predicates=(_predicate("agent:decided_in", "agent"),),
    )

    loaded = load_packs([code, agent])

    assert set(loaded.concepts) == {"code:function", "agent:decision"}
    assert set(loaded.predicates) == {"code:calls", "agent:decided_in"}
    assert loaded.concepts["agent:decision"].pack == "agent"


def test_contested_concept_uri_names_both_packs_and_the_key() -> None:
    first = _StubPack(name="code", _concepts=(_concept("shared:thing", "code"),))
    second = _StubPack(name="agent", _concepts=(_concept("shared:thing", "agent"),))

    with pytest.raises(PackConflictError) as raised:
        load_packs([first, second])

    assert raised.value.key == "shared:thing"
    assert (raised.value.first_pack, raised.value.second_pack) == ("code", "agent")


def test_contested_predicate_id_is_refused_too() -> None:
    first = _StubPack(name="code", _predicates=(_predicate("shared:rel", "code"),))
    second = _StubPack(name="agent", _predicates=(_predicate("shared:rel", "agent"),))

    with pytest.raises(PackConflictError, match="shared:rel"):
        load_packs([first, second])


def test_a_predicate_conflict_leaves_no_pack_loaded() -> None:
    first = _StubPack(
        name="code",
        _concepts=(_concept("code:function", "code"),),
        _predicates=(_predicate("shared:rel", "code"),),
    )
    second = _StubPack(name="agent", _predicates=(_predicate("shared:rel", "agent"),))

    with pytest.raises(PackConflictError):
        load_packs([first, second])


def test_concepts_are_iterated_lazily_not_materialised() -> None:
    pack = _StubPack(name="code", _concepts=(_concept("code:function", "code"),))
    assert pack.seen == []

    load_packs([pack])

    assert pack.seen == ["code:function"]
