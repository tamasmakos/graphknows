"""The identity ladder: what each rung decides, and in what order (FR-016..FR-021)."""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.consolidation.entity_resolution import (
    EntityView,
    Layer,
    Resolver,
    mention_layer,
)
from graphknows.models.fact import Mention

pytestmark = pytest.mark.unit


def _mention(surface: str, label: str, segment_id: str = "s1") -> Mention:
    return Mention(
        segment_id=segment_id,
        entity_id=surface.casefold(),
        surface=surface,
        span=(0, len(surface)),
        label=label,
    )


def _view(entity_id: str, **fields: Any) -> EntityView:
    return EntityView(id=entity_id, name_norm=entity_id, **fields)


# --- the mention layer ------------------------------------------------------


def test_surfaces_survive_as_surfaces() -> None:
    """A surface form is evidence, never an alias string on the entity (FR-016)."""
    views = mention_layer([_mention("Acme", "org"), _mention("ACME", "org", "s2")])
    assert views["acme"].surfaces == ("Acme", "ACME")


def test_type_is_a_histogram_not_the_first_label() -> None:
    """Conflicting types are tallied, not fixed at first sight (FR-021)."""
    mentions = [
        _mention("run", "function"),
        _mention("run", "module", "s2"),
        _mention("run", "function", "s3"),
    ]
    assert mention_layer(mentions)["run"].type_histogram == {"function": 2, "module": 1}


# --- the ladder -------------------------------------------------------------


def test_veto_beats_strong_evidence() -> None:
    """A pack veto stops a pair that every other signal would merge (FR-019)."""
    resolver = Resolver(veto=lambda a, b: True)
    verdict = resolver.verdict(_view("run"), _view("run"))
    assert verdict.layer is Layer.VETO


def test_disjoint_siblings_produce_neither_merge_nor_candidate() -> None:
    left = _view("mercury", concepts=frozenset({"planet"}))
    right = EntityView(id="m2", name_norm="mercury", concepts=frozenset({"metal"}))
    resolver = Resolver(disjoint=frozenset({frozenset({"planet", "metal"})}))
    plan = resolver.plan({"mercury": left, "m2": right}, {"mercury": ["m2"]})
    assert (plan.merges, plan.candidates, plan.vetoed) == ((), (), 1)


def test_identical_key_in_one_block_merges() -> None:
    """Strong evidence: same normalised key, same block, agreeing concepts (FR-017)."""
    left = _view("acme", type_histogram={"org": 2}, concepts=frozenset({"org"}))
    right = _view("acme", type_histogram={"org": 1}, concepts=frozenset({"org"}))
    assert Resolver().verdict(left, right).layer is Layer.STRONG


def test_a_different_block_is_not_strong() -> None:
    """The same key read under a different dominant type is a different block."""
    left = _view("run", type_histogram={"function": 2})
    right = _view("run", type_histogram={"module": 2})
    assert Resolver().verdict(left, right).layer is not Layer.STRONG


def test_shared_concept_is_only_a_candidate() -> None:
    """Weaker evidence defers commitment to recall (FR-018)."""
    left = _view("acme", concepts=frozenset({"org"}))
    right = _view("acme corporation", concepts=frozenset({"org"}))
    verdict = Resolver().verdict(left, right)
    assert verdict.layer is Layer.WEAK and verdict.score >= Resolver().floor


def test_similarity_must_clear_floor_plus_margin() -> None:
    near = _view("a", embedding=(1.0, 0.0))
    far = _view("b", embedding=(0.0, 1.0))
    assert Resolver().verdict(near, _view("c", embedding=(1.0, 0.0))).layer is Layer.WEAK
    assert Resolver().verdict(near, far).layer is Layer.NONE


# --- the plan ---------------------------------------------------------------


def test_merge_log_is_complete_before_any_structural_change() -> None:
    """Every merge is a log entry carrying layer, evidence and time (FR-017).

    Planning is pure: the views it read are untouched, so the ``MERGED_INTO``
    log for the whole batch exists before a single merge is applied — which is
    what makes the merge replayable, and reversible, from the log alone.
    """
    views = {
        "acme": _view("acme", type_histogram={"org": 3}),
        "acme2": EntityView(id="acme2", name_norm="acme", type_histogram={"org": 1}),
    }
    plan = Resolver().plan(views, {"acme": ["acme2"]})
    assert len(plan.merges) == 1
    log = plan.merges[0]
    assert (log.source_id, log.target_id, log.layer) == ("acme2", "acme", Layer.STRONG)
    assert log.evidence == "key=acme" and log.at is not None
    assert views["acme"].type_histogram == {"org": 3}


def test_a_pair_is_decided_once() -> None:
    views = {
        "acme": _view("acme", type_histogram={"org": 3}),
        "acme2": EntityView(id="acme2", name_norm="acme", type_histogram={"org": 1}),
    }
    plan = Resolver().plan(views, {"acme": ["acme2"], "acme2": ["acme"]})
    assert len(plan.merges) == 1


def test_plan_reports_its_counters() -> None:
    views = {
        "acme": _view("acme", type_histogram={"org": 3}),
        "acme2": EntityView(id="acme2", name_norm="acme", type_histogram={"org": 1}),
        "zeta": _view("zeta"),
    }
    plan = Resolver(veto=lambda a, b: "zeta" in (a, b)).plan(
        views, {"acme": ["acme2", "zeta"], "acme2": ["zeta"]}
    )
    counters = plan.counters
    assert (counters.merges_committed, counters.merges_vetoed) == (1, 2)
    assert counters.candidates_truncated == 0
