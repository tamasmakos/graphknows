"""Unit tests for the relation verifier gate (no model download).

The DeBERTa checkpoint is never loaded here: the null gate is a pure
pass-through, and the active gate is exercised with a stub standing in for the
loaded ``RelationVerifier`` so the offset-recovery and drop logic can be checked
deterministically.
"""

from __future__ import annotations

from graphknows.ingestion.extraction.relations.verifier import (
    NullRelationVerifier,
    RelationVerifierGate,
    build_relation_verifier,
)


def _rel(head: str, relation: str, tail: str) -> dict:
    return {"head": head, "relation": relation, "tail": tail, "source": "relex"}


def test_null_verifier_passes_relations_through():
    rels = [_rel("Jon", "WORK_ON", "dance routines")]
    assert NullRelationVerifier().filter("Jon works on dance routines.", rels) is rels


def test_build_returns_null_when_disabled():
    class S:
        relation_verifier = False
        relation_verifier_model = "x"
        relation_verifier_threshold = 0.5

    assert isinstance(build_relation_verifier(S()), NullRelationVerifier)


class _StubVerifier:
    """Stands in for a loaded RelationVerifier: keeps a fixed surface pair."""

    def __init__(self, keep: set[tuple[str, str]]) -> None:
        self._keep = keep

    def verify(self, text: str, grouped: dict) -> dict:
        out: dict = {}
        for rel_type, insts in grouped.items():
            survivors = []
            for inst in insts:
                pair = (inst["head"]["text"], inst["tail"]["text"])
                if pair in self._keep:
                    inst = {**inst, "verifier_score": 0.9}
                    survivors.append(inst)
            out[rel_type] = survivors
        return out


def _gate_with(stub: _StubVerifier) -> RelationVerifierGate:
    gate = RelationVerifierGate("stub", 0.5)
    gate._verifier = stub  # bypass lazy load
    return gate


def test_gate_drops_unverified_and_recovers_offsets():
    text = "Jon works on dance routines but did not tackle business goals."
    rels = [
        _rel("Jon", "WORK_ON", "dance routines"),
        _rel("dance routines", "TACKLE", "business goals"),
    ]
    gate = _gate_with(_StubVerifier(keep={("Jon", "dance routines")}))

    kept = gate.filter(text, rels)

    assert [r["relation"] for r in kept] == ["WORK_ON"]
    assert kept[0]["verifier_score"] == 0.9


def test_gate_keeps_unlocatable_relations():
    """A surface absent from the text is unverifiable, not disproven — keep it."""
    text = "Jon works on dance routines."
    rels = [_rel("Gina", "OWNS", "a store")]  # neither surface appears verbatim
    gate = _gate_with(_StubVerifier(keep=set()))

    assert gate.filter(text, rels) == rels


def test_gate_load_failure_passes_through():
    """If the model cannot load, the gate must not delete relations."""
    gate = RelationVerifierGate("does-not-exist/nope", 0.5)
    rels = [_rel("Jon", "WORK_ON", "dance routines")]

    assert gate.filter("Jon works on dance routines.", rels) == rels
    assert gate._disabled is True
