"""There is no FE plane (issue #143).

Today ``framenet._core_fes`` keeps only ``coreType == "Core"`` and
``frame_srl.frame_ontology`` buckets every other frame element into
``peripheral_fes``. FrameNet 1.7 ships four coreness values, plus
``requiresFE``/``excludesFE``/``FEcoreSets``/``semType`` metadata, and none of
it reaches the graph or gates a single frame instance: a role configuration
that violates FrameNet's own Requires/Excludes constraints, or that double-fills
one CoreSet while leaving another empty, is written to the graph as fact, and
an unfilled core role is never reported at all.

These tests pin the missing plane (``graphknows.symbolic.framenet.fe``) and the
validity/fillability gating it should drive in
``_STMIngestHandler._frame_instances``. All corpus numbers below are verified
against the shipped FrameNet 1.7 corpus (see the docstring literals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.relations import frame_srl
from graphknows.ingestion.extraction.relations.frame_srl import FrameInstance
from graphknows.ingestion.stm.ingest import _STMIngestHandler

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Section 1: the pure FE plane (graphknows.symbolic.framenet.fe)
# ---------------------------------------------------------------------------


def test_coreness_is_stored_in_full_and_exhaustive_over_the_corpus() -> None:
    from graphknows.symbolic.framenet import fe

    assert set(fe.CORENESS) == {"Core", "Peripheral", "Extra-Thematic", "Core-Unexpressed"}
    # Exhaustive-with-guard, same pattern as relations.unknown_relation_types():
    # a coreness value the corpus ships that this table doesn't name must
    # surface, not be swept into a default bucket.
    assert fe.unknown_coreness() == set()


def test_core_unexpressed_is_not_conflated_with_core_or_peripheral() -> None:
    """Attempt/Goal is Core-Unexpressed — a real gap in _core_fes today, which
    only keeps plain 'Core' and would silently drop this FE."""
    from graphknows.symbolic.framenet import fe

    goal = fe.frame_elements("Attempt")["Goal"]
    assert goal.coreness == "Core-Unexpressed"


def test_requires_and_excludes_are_read_from_the_corpus() -> None:
    from graphknows.symbolic.framenet import fe

    assert fe.frame_elements("Attaching")["Item"].requires == "Goal"
    assert fe.frame_elements("Aiming")["Targeted"].excludes == "Target_location"


def test_core_sets_partition_aiming_and_are_absent_for_attaching() -> None:
    from graphknows.symbolic.framenet import fe

    assert fe.core_sets("Aiming") == {
        frozenset({"Instrument", "Activity", "Agent"}),
        frozenset({"Target_location", "Targeted"}),
    }
    assert fe.core_sets("Attaching") == set()


def test_violation_flags_unmet_requires() -> None:
    from graphknows.symbolic.framenet import fe

    reason = fe.violation("Attaching", {"Item", "Agent"})
    assert reason
    assert "Item" in reason and "Goal" in reason
    assert fe.violation("Attaching", {"Item", "Goal"}) is None


def test_violation_flags_excludes_conflicts() -> None:
    from graphknows.symbolic.framenet import fe

    reason = fe.violation("Aiming", {"Targeted", "Target_location"})
    assert reason
    assert "Target_location" in reason
    assert fe.violation("Aiming", {"Targeted"}) is None


def test_core_sets_satisfied_requires_a_fill_from_every_set() -> None:
    from graphknows.symbolic.framenet import fe

    # Agent + Instrument both come from the SAME core set; Target_location/
    # Targeted is left wholly empty.
    assert fe.core_sets_satisfied("Aiming", {"Agent", "Instrument"}) is False
    # Agent covers the first set, Targeted covers the second.
    assert fe.core_sets_satisfied("Aiming", {"Agent", "Targeted"}) is True
    # A frame with no core sets is vacuously satisfied.
    assert fe.core_sets_satisfied("Attaching", {"Item"}) is True


def _find_sentient_core_fe() -> tuple[str, str]:
    """A real (frame, FE) pair whose FE is Core with semType Sentient/Human.

    Discovered from the shipped rows rather than hard-coded, per the task.
    """
    from graphknows.symbolic.framenet import fe

    for row in fe.fe_rows():
        if row.coreness == "Core" and row.sem_type in ("Sentient", "Human"):
            return row.frame, row.name
    raise AssertionError("no Core FE with semType Sentient/Human found in the shipped corpus")


def _find_unconstrained_fe() -> tuple[str, str]:
    """A real (frame, FE) pair whose semType exists but maps to no entity type.

    Discovered from the shipped rows rather than hard-coded, per the task.
    """
    from graphknows.symbolic.framenet import fe

    for row in fe.fe_rows():
        if row.sem_type and row.sem_type not in fe._SEMTYPE_ALLOWED:
            return row.frame, row.name
    raise AssertionError("no unconstrained semType FE found in the shipped corpus")


def test_semtype_conflict_is_a_pure_predicate_over_filler_type() -> None:
    from graphknows.symbolic.framenet import fe

    frame, role = _find_sentient_core_fe()
    unconstrained_frame, unconstrained_role = _find_unconstrained_fe()
    # Unknown filler type never conflicts — absence of evidence is not evidence.
    assert fe.semtype_conflict(frame, role, "") is False
    # A semtype the mapping does not constrain never conflicts.
    assert fe.semtype_conflict(unconstrained_frame, unconstrained_role, "PERSON") is False
    # A place cannot be a sentient being; a person can.
    assert fe.semtype_conflict(frame, role, "LOCATION") is True
    assert fe.semtype_conflict(frame, role, "PERSON") is False


def test_plane_row_counts_and_histogram_match_the_shipped_corpus() -> None:
    from graphknows.symbolic.framenet import fe

    rows = fe.fe_rows()
    assert len(rows) == 11428
    assert len(fe.semtype_rows()) == 109

    from collections import Counter

    hist = Counter(r.coreness for r in rows)
    assert hist == {
        "Core": 3501,
        "Peripheral": 5023,
        "Extra-Thematic": 2781,
        "Core-Unexpressed": 123,
    }


# ---------------------------------------------------------------------------
# Section 2: gating in _STMIngestHandler._frame_instances
# ---------------------------------------------------------------------------


class _FakeMorph:
    def get(self, _key: str) -> list[Any]:
        return []


class _FakeToken:
    """Just enough of a spaCy Token for graphknows.linguistics.analyse()."""

    def __init__(self, idx: int, text: str, dep_: str = "ROOT") -> None:
        self.idx = idx
        self.text = text
        self.dep_ = dep_
        self.pos_ = "AUX" if dep_ == "auxpass" else "VERB"
        self.lemma_ = text.lower()
        self.i = idx
        self.children: list[_FakeToken] = []
        self.head = self
        self.morph = _FakeMorph()
        self.sent = [self]


def _fake_doc(passive: bool) -> list[_FakeToken]:
    verb = _FakeToken(idx=0, text="filed")
    if passive:
        verb.children = [_FakeToken(idx=10, text="was", dep_="auxpass")]
    return [verb]


class _StubExtractor:
    def __init__(self, doc: list[_FakeToken]) -> None:
        self.nlp = object()
        self.gliner = object()
        self._doc = doc

    def parse(self, _text: str) -> list[_FakeToken]:
        return self._doc


def _make_handler(passive: bool = False) -> _STMIngestHandler:
    store = MagicMock()
    for attr in ("write_frame_instances", "write_frame_tags"):
        setattr(store, attr, AsyncMock(return_value=None))
    extractor = _StubExtractor(_fake_doc(passive))
    return _STMIngestHandler(store, extractor=extractor, embed_fn=MagicMock())


@dataclass
class _Hit:
    """Duck-typed FrameHit — content is irrelevant once extract_frame_instances_srl
    is monkeypatched; only its presence forces the SRL branch."""

    frame: str = "Aiming"
    offset: int = 0
    trigger: str = "aimed"


def _patch_srl(monkeypatch: pytest.MonkeyPatch, instances: list[FrameInstance]) -> None:
    monkeypatch.setattr(
        frame_srl,
        "extract_frame_instances_srl",
        lambda *a, **k: instances,
    )


def test_excludes_violation_drops_the_instance_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _make_handler()
    inst = FrameInstance(
        frame="Aiming",
        roles={"Targeted": ["London"], "Target_location": ["Berlin"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [inst])

    out, _nesting = handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])

    assert out == []
    assert any("exclude" in k.lower() and v > 0 for k, v in handler._abstentions.items())


def test_requires_violation_drops_the_instance_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _make_handler()
    inst = FrameInstance(
        frame="Attaching",
        roles={"Item": ["rope"], "Agent": ["Marta"]},  # Goal absent
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [inst])

    out, _nesting = handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])

    assert out == []
    assert any("require" in k.lower() and v > 0 for k, v in handler._abstentions.items())


def test_double_filling_one_core_set_drops_the_instance_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _make_handler()
    # Agent + Instrument are both drawn from Aiming's {Instrument, Activity,
    # Agent} core set, leaving {Target_location, Targeted} wholly unfilled.
    bad = FrameInstance(
        frame="Aiming",
        roles={"Agent": ["Marta"], "Instrument": ["bow"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [bad])
    out, _nesting = handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])
    assert out == []
    assert any(
        "core_set" in k.lower() or "coreset" in k.lower()
        for k, v in handler._abstentions.items()
        if v > 0
    )


def test_one_role_per_core_set_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    # Agent covers the first core set, Targeted covers the second: both sets
    # satisfied, no Requires/Excludes conflict.
    good = FrameInstance(
        frame="Aiming",
        roles={"Agent": ["Marta"], "Targeted": ["London"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [good])
    out, _nesting = handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])
    assert len(out) == 1
    assert out[0]["frame"] == "Aiming"


def _sentient_core_role_and_partner() -> tuple[str, str, str]:
    """A (frame, sentient-core-role, safe-partner-role) triple with no other
    Requires/Excludes/CoreSet conflict, discovered from the live plane rather
    than hard-coded."""
    from graphknows.symbolic.framenet import fe

    for row in fe.fe_rows():
        if row.coreness != "Core" or row.sem_type not in ("Sentient", "Human"):
            continue
        frame, role = row.frame, row.name
        for other in fe.frame_elements(frame).values():
            if other.name == role or other.coreness not in ("Core", "Core-Unexpressed"):
                continue
            filled = {role, other.name}
            if fe.violation(frame, filled) is None and fe.core_sets_satisfied(frame, filled):
                return frame, role, other.name
    raise AssertionError("no usable Sentient/Human Core FE pairing found in the shipped corpus")


def test_semtype_gate_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, role, partner = _sentient_core_role_and_partner()
    handler = _make_handler()
    inst = FrameInstance(
        frame=frame,
        roles={role: ["Berlin"], partner: ["Elena"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [inst])

    out, _nesting = handler._frame_instances(
        "text", "chunk1", "speaker", None, [_Hit()], {"berlin": "LOCATION"}
    )

    assert len(out) == 1
    assert not any("semtype" in k.lower() or "type_gate" in k.lower() for k in handler._abstentions)


def test_semtype_gate_vetoes_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_ENABLE_FE_TYPE_GATE", "true")
    frame, role, partner = _sentient_core_role_and_partner()
    handler = _make_handler()
    inst = FrameInstance(
        frame=frame,
        roles={role: ["Berlin"], partner: ["Elena"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )
    _patch_srl(monkeypatch, [inst])

    out, _nesting = handler._frame_instances(
        "text", "chunk1", "speaker", None, [_Hit()], {"berlin": "LOCATION"}
    )

    assert out == []
    assert any(
        ("semtype" in k.lower() or "type_gate" in k.lower()) and v > 0
        for k, v in handler._abstentions.items()
    )


def test_unfilled_core_reporting_splits_cni_from_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passive clauses licence a CNI (Constructional Null Instantiation) reading
    of an unfilled core role; active ones do not — the omission is unexplained."""
    passive_handler = _make_handler(passive=True)
    active_handler = _make_handler(passive=False)

    # Attaching/Goal left unfilled in both cases (Item + Agent filled).
    inst = FrameInstance(
        frame="Attaching",
        roles={"Item": ["rope"], "Agent": ["Marta"]},
        match_sim=1.0,
        role_confidence=0.9,
        trigger_offset=0,
    )

    monkeypatch.setattr(frame_srl, "extract_frame_instances_srl", lambda *a, **k: [inst])
    passive_handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])
    active_handler._frame_instances("text", "chunk1", "speaker", None, [_Hit()])

    passive_cni = sum(v for k, v in passive_handler._abstentions.items() if k.endswith("cni"))
    passive_unknown = sum(
        v for k, v in passive_handler._abstentions.items() if k.endswith("unknown")
    )
    active_cni = sum(v for k, v in active_handler._abstentions.items() if k.endswith("cni"))
    active_unknown = sum(v for k, v in active_handler._abstentions.items() if k.endswith("unknown"))

    assert passive_cni > 0 and passive_unknown == 0
    assert active_unknown > 0 and active_cni == 0
