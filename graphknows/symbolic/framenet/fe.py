"""The FrameNet frame-element (FE) plane — coreness, validity, fillability.

FrameNet 1.7 ships far more about a frame element than Core/not-Core, and
none of it reached the graph before this module: ``framenet._core_fes`` kept
only ``coreType == "Core"`` and ``frame_srl.frame_ontology`` bucketed every
other FE into ``peripheral_fes``, discarding the rest.

Verified against the shipped FrameNet 1.7 corpus (``nltk.corpus.framenet``):

- 1221 frames, 11428 frame elements.
- Coreness histogram: Core 3501, Peripheral 5023, Extra-Thematic 2781,
  Core-Unexpressed 123.
- 105 FEs declare ``requiresFE``, 455 declare ``excludesFE`` — each at most
  one (the corpus never lists more than a single requires/excludes target for
  a given FE), so both are modelled as single strings, not lists.
- 381 ``FEcoreSets`` over the 1221 frames (most frames have none).
- 109 semantic types (``fn.semtypes()``).

Deliberately NOT called: ``fn.propagate_semtypes()``. That loads every frame
file to push semantic types down the inheritance hierarchy and takes seconds;
the annotation this module reads is the sparse, as-shipped one, which is what
the issue asks for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

_MARKUP = re.compile(r"<[^>]+>")

# The four coreness values FrameNet 1.7 declares. Core-Unexpressed is a CORE
# role that this particular frame cannot express locally -- it is inherited
# or bound elsewhere (e.g. Attempt.Goal, bound by the frame Attempt is used
# in) -- which is why it must not be lumped in with Peripheral the way
# frame_srl.frame_ontology does today, and must not be counted as a fillable
# core role either: it is neither absent nor locally satisfiable.
CORENESS: frozenset[str] = frozenset({"Core", "Peripheral", "Extra-Thematic", "Core-Unexpressed"})

# The graph's own entity-type vocabulary (graphknows.ingestion.extraction.
# entities.extractor._SUPERSENSE_TO_TYPE's value set) that a FrameNet semantic
# type has a defensible equivalent in. Every other semantic type (Manner,
# Degree, Message, Content, Speed, Quantity, ...) constrains nothing -- that
# is why semtype_conflict is an experiment and ships default-off.
_SEMTYPE_ALLOWED: dict[str, frozenset[str]] = {
    "Sentient": frozenset({"PERSON", "ANIMAL"}),
    "Human": frozenset({"PERSON", "ANIMAL"}),
    "Animate_being": frozenset({"PERSON", "ANIMAL"}),
    "Living_thing": frozenset({"PERSON", "ANIMAL"}),
    "Physical_object": frozenset({"ARTIFACT", "PRODUCT", "POSSESSION"}),
    "Artifact": frozenset({"ARTIFACT", "PRODUCT", "POSSESSION"}),
    "Container": frozenset({"ARTIFACT", "PRODUCT", "POSSESSION"}),
    "Physical_entity": frozenset({"ARTIFACT", "PRODUCT", "POSSESSION"}),
    "Location": frozenset({"LOCATION"}),
    "Locative_relation": frozenset({"LOCATION"}),
    "Path": frozenset({"LOCATION"}),
    "Source": frozenset({"LOCATION"}),
    "Goal": frozenset({"LOCATION"}),
    "Time": frozenset({"TIME"}),
    "Duration": frozenset({"TIME"}),
    "Group": frozenset({"ORGANIZATION", "PERSON"}),
    "Organization": frozenset({"ORGANIZATION", "PERSON"}),
    "Event": frozenset({"EVENT", "ACTIVITY"}),
    "Human_act": frozenset({"EVENT", "ACTIVITY"}),
    "Activity": frozenset({"EVENT", "ACTIVITY"}),
    "State_of_affairs": frozenset({"EVENT", "ACTIVITY"}),
}


def _clean(text: str | None) -> str:
    return _MARKUP.sub("", text or "").strip()


@dataclass(frozen=True)
class FrameElement:
    """One frame element, with the metadata Core/not-Core used to discard."""

    name: str
    coreness: str
    definition: str
    semtype: str = ""
    requires: str = ""
    excludes: str = ""


@dataclass(frozen=True)
class _FeRow:
    """One FE as a flat write row for the storage layer (see fe_rows())."""

    id: str
    frame: str
    name: str
    coreness: str
    definition: str
    sem_type: str
    requires: str
    excludes: str
    core_set: int


@dataclass(frozen=True)
class _SemtypeRow:
    """One FrameNet semantic type as a flat write row (see semtype_rows())."""

    name: str
    abbrev: str
    definition: str


@lru_cache(maxsize=1221)
def frame_elements(frame: str) -> dict[str, FrameElement]:
    """Frame elements of *frame*, keyed by name.

    Empty dict on an unknown frame name -- the corpus raises on a lookup
    miss; frame_srl.frame_ontology already takes this stance.
    """
    from graphknows.symbolic.framenet.framenet import _framenet

    try:
        f = _framenet().frame(frame)
    except Exception:
        return {}
    return {
        name: FrameElement(
            name=name,
            coreness=o.coreType,
            definition=_clean(o.definition)[:400],
            semtype=o.semType.name if o.semType else "",
            requires=o.requiresFE.name if o.requiresFE else "",
            excludes=o.excludesFE.name if o.excludesFE else "",
        )
        for name, o in f.FE.items()
    }


@lru_cache(maxsize=1221)
def core_sets(frame: str) -> frozenset[frozenset[str]]:
    """This frame's FEcoreSets -- alternative fillers for one core participant.

    Empty for an unknown frame or one with none declared (most frames: 381
    sets over 1221 frames).
    """
    from graphknows.symbolic.framenet.framenet import _framenet

    try:
        f = _framenet().frame(frame)
    except Exception:
        return frozenset()
    return frozenset(frozenset(o.name for o in cs) for cs in f.FEcoreSets)


def violation(frame: str, filled: Iterable[str]) -> str | None:
    """The first Requires/Excludes constraint *filled* breaks, or None.

    For every filled FE: its ``requires`` target must also be filled, and its
    ``excludes`` target must not be. Exclusion is treated as SYMMETRIC (the
    union of both declared directions) because FrameNet does not always
    declare it on both FEs of a pair, and mutual exclusion is symmetric by
    definition -- Aiming.Targeted excludes Target_location even though
    Target_location declares no excludesFE back.
    """
    fes = frame_elements(frame)
    filled_set = set(filled)
    for name in filled_set:
        fe = fes.get(name)
        if fe is None:
            continue
        if not fe.requires or fe.requires in filled_set:
            continue
        target = fes.get(fe.requires)
        # A Core-Unexpressed target is never locally fillable (bound elsewhere
        # per FrameNet's own semantics -- see the CORENESS docstring), so
        # requiring it to be filled here can never be satisfied and is not a
        # real violation.
        if target is not None and target.coreness == "Core-Unexpressed":
            continue
        return f"requires:{name}->{fe.requires}"
    excl_pairs = {frozenset({name, fe.excludes}) for name, fe in fes.items() if fe.excludes}
    for pair in excl_pairs:
        if pair <= filled_set:
            a, b = sorted(pair)
            return f"excludes:{a}|{b}"
    return None


def core_sets_satisfied(frame: str, filled: Iterable[str]) -> bool:
    """True when every core set of *frame* has at least one filled member.

    Vacuously True for a frame with no core sets. A core set's members are
    alternative ways of expressing the same core participant, so two fills
    from one set are one participant, not two -- this is what lets a
    fillability gate distinguish that from a genuinely unfilled participant.

    A core set none of whose members are coreness ``Core`` (all
    Core-Unexpressed) is also vacuously satisfied: the extraction schema only
    ever offers plain-Core FEs (frame_srl.frame_ontology), so such a set can
    never be filled locally and demanding a fill from it would drop every
    instance of the frame unconditionally (e.g. Being_at_risk, whose only
    core set is {Dangerous_entity, Harmful_event}, both Core-Unexpressed).
    """
    filled_set = set(filled)
    fes = frame_elements(frame)
    for cs in core_sets(frame):
        if not any(fes[name].coreness == "Core" for name in cs if name in fes):
            continue
        if not cs & filled_set:
            return False
    return True


def semtype_conflict(frame: str, role: str, entity_type: str) -> bool:
    """The default-off type veto's predicate: pure, no settings lookup.

    False whenever *entity_type* is empty/unknown, whenever the FE has no
    semantic type, or whenever that semantic type is not in the small
    equivalence table above. Otherwise True iff *entity_type* falls outside
    the allowed set.
    """
    if not entity_type:
        return False
    fe = frame_elements(frame).get(role)
    if fe is None or not fe.semtype:
        return False
    allowed = _SEMTYPE_ALLOWED.get(fe.semtype)
    if allowed is None:
        return False
    return entity_type not in allowed


@lru_cache(maxsize=1)
def unknown_coreness() -> frozenset[str]:
    """Coreness strings the corpus declares that CORENESS does not name.

    Mirrors relations.unknown_relation_types(): non-empty means a future
    FrameNet release added a coreness value, reported rather than swallowed.
    """
    from graphknows.symbolic.framenet.framenet import _framenet

    seen = {o.coreType for f in _framenet().frames() for o in f.FE.values()}
    return frozenset(seen - CORENESS)


@lru_cache(maxsize=1)
def fe_rows() -> tuple[_FeRow, ...]:
    """Every frame element in the corpus as a flat write row.

    Built over ``frame_elements()``/``core_sets()`` rather than re-walking the
    corpus a second time, so the coreType/definition/semType/requires/excludes
    extraction is done in exactly one place.

    ``graphknows.storage`` sits below ``graphknows.symbolic`` in the
    import-linter layer contract and cannot import this module, so the rows
    are built here and handed down -- the same shape as ``write_ontology``
    taking a duck-typed index today. 11428 rows on the shipped corpus.
    """
    from graphknows.symbolic.framenet.framenet import _framenet

    rows: list[_FeRow] = []
    for f in _framenet().frames():
        set_of = {name: i for i, cs in enumerate(core_sets(f.name)) for name in cs}
        for name, elem in frame_elements(f.name).items():
            rows.append(
                _FeRow(
                    id=f"{f.name}:{name}",
                    frame=f.name,
                    name=name,
                    coreness=elem.coreness,
                    definition=elem.definition,
                    sem_type=elem.semtype,
                    requires=elem.requires,
                    excludes=elem.excludes,
                    core_set=set_of.get(name, -1),
                )
            )
    return tuple(rows)


@lru_cache(maxsize=1)
def semtype_rows() -> tuple[_SemtypeRow, ...]:
    """Every FrameNet semantic type as a flat write row. 109 on the shipped corpus."""
    from graphknows.symbolic.framenet.framenet import _framenet

    return tuple(
        _SemtypeRow(name=st.name, abbrev=st.abbrev, definition=_clean(st.definition)[:400])
        for st in _framenet().semtypes()
    )
