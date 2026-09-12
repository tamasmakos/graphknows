"""The symbol vocabulary: three kinds, and a definition that must be embeddable."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphknows.models import ConceptRef, PredicateRef, SymbolKind


def test_symbol_kinds_are_concept_predicate_and_episode() -> None:
    assert {kind.value for kind in SymbolKind} == {"concept", "predicate", "episode"}


def test_concept_carries_its_definition_and_pack() -> None:
    concept = ConceptRef(
        uri="code:function",
        label="function",
        definition="A named callable unit of code.",
        pack="code",
    )
    assert concept.pack == "code"
    assert concept.definition.startswith("A named callable")


def test_predicate_is_non_functional_and_unconstrained_by_default() -> None:
    predicate = PredicateRef(
        id="code:calls",
        label="calls",
        definition="The subject invokes the object.",
        canonical="calls",
        pack="code",
    )
    assert predicate.functional is False
    assert predicate.domain is None
    assert predicate.range is None


@pytest.mark.parametrize("symbol_type", [ConceptRef, PredicateRef])
def test_blank_definition_is_rejected(symbol_type: type) -> None:
    fields = {
        "uri": "code:thing",
        "id": "code:thing",
        "label": "thing",
        "definition": "   ",
        "canonical": "thing",
        "pack": "code",
    }
    with pytest.raises(ValidationError, match="definition is required"):
        symbol_type(**{k: v for k, v in fields.items() if k in symbol_type.model_fields})


def test_symbols_are_frozen() -> None:
    concept = ConceptRef(uri="code:x", label="x", definition="An x.", pack="code")
    with pytest.raises(ValidationError):
        concept.label = "y"  # type: ignore[misc]
