"""Which relation properties reach the extractor (#137).

Two things are pinned here:

* ``_allocate_offer`` — one slot per (domain, range) pair as a floor, remainder
  shared in proportion to group size. The floor preserves the original
  round-robin's intent; the proportion is what a flat round-robin got backwards.
* The bundled personal profile is shipped inside the package and reachable by
  name, so the CCO-vs-profile question is an A/B rather than a fixture only the
  test suite can load.
"""

from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path

import pytest

from graphknows.ingestion.stm.ingest import _RELATION_LABEL_CAP, _allocate_offer
from graphknows.settings import GraphKnowsSettings

pytestmark = pytest.mark.unit

PACKAGE_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "graphknows"
    / "symbolic"
    / "ontology"
    / "assets"
    / "personal"
    / "personal-profile.json"
)
FIXTURE_PROFILE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ontology" / "personal-profile.json"
)


class _Prop:
    """Minimal stand-in for an OntologyTerm — only what the allocator reads."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Prop({self.label!r})"


def _groups(**sizes: int) -> dict[tuple[tuple[str, ...], tuple[str, ...]], list[_Prop]]:
    return {
        ((name,), (name,)): [_Prop(f"{name}{i}") for i in range(n)] for name, n in sizes.items()
    }


class TestAllocateOffer:
    def test_every_group_keeps_at_least_one_slot(self) -> None:
        """The original round-robin's intent: no pair is deleted from the offer."""
        offer = _allocate_offer(_groups(big=50, small=1, tiny=1), cap=10)
        assert {p.label[:-1] if p.label[-1].isdigit() else p.label for p in offer} >= {
            "small",
            "tiny",
        }

    def test_the_dominant_group_gets_most_of_the_remainder(self) -> None:
        """A flat round-robin gave a 50-member group the same share as a 1-member one."""
        offer = _allocate_offer(_groups(big=50, small=1, tiny=1), cap=10)
        big = sum(1 for p in offer if p.label.startswith("big"))
        assert big >= 6, f"dominant group starved: only {big} of {len(offer)} slots"

    def test_the_cap_is_respected(self) -> None:
        assert len(_allocate_offer(_groups(a=50, b=50, c=50), cap=7)) == 7

    def test_more_groups_than_slots_keeps_the_largest(self) -> None:
        offer = _allocate_offer(_groups(a=9, b=5, c=1, d=1, e=1), cap=2)
        assert len(offer) == 2
        assert {p.label for p in offer} == {"a0", "b0"}

    def test_no_groups_offers_nothing(self) -> None:
        assert _allocate_offer({}, cap=10) == []

    def test_a_small_vocabulary_is_offered_whole(self) -> None:
        """Nothing is dropped when everything applicable fits inside the cap."""
        groups = _groups(x=9, y=4, z=3)
        offer = _allocate_offer(groups, cap=_RELATION_LABEL_CAP)
        assert len(offer) == 16

    def test_within_group_order_survives(self) -> None:
        """An embedding rank applied upstream still decides who is kept."""
        offer = _allocate_offer(_groups(only=5), cap=3)
        assert [p.label for p in offer] == ["only0", "only1", "only2"]


class TestStarvationRegression:
    """The measured defect: 87 familial predicates available, 5 offered."""

    def test_flat_round_robin_starves_the_relevant_group(self) -> None:
        groups = _groups(familial=87, mereology=4, temporal=3, other=5)
        flat = [p for row in zip_longest(*groups.values()) for p in row if p is not None][:15]
        allocated = _allocate_offer(groups, _RELATION_LABEL_CAP)

        flat_fam = sum(1 for p in flat if p.label.startswith("familial"))
        alloc_fam = sum(1 for p in allocated if p.label.startswith("familial"))
        assert flat_fam <= 5, "fixture no longer reproduces the flat round-robin's starvation"
        assert alloc_fam > flat_fam * 2, (
            f"proportional allocation did not fix the starvation: {alloc_fam} vs {flat_fam}"
        )


class TestBundledProfile:
    def test_the_profile_ships_inside_the_package(self) -> None:
        """A test fixture cannot be loaded from an installed wheel."""
        assert PACKAGE_PROFILE.is_file()

    def test_the_package_copy_has_not_drifted_from_the_fixture(self) -> None:
        assert json.loads(PACKAGE_PROFILE.read_text(encoding="utf-8")) == json.loads(
            FIXTURE_PROFILE.read_text(encoding="utf-8")
        )

    def test_the_whole_profile_fits_inside_the_label_cap(self) -> None:
        """Its design intent is to be offered WHOLE, with no selection step."""
        profile = json.loads(PACKAGE_PROFILE.read_text(encoding="utf-8"))
        assert len(profile["properties"]) <= _RELATION_LABEL_CAP

    def test_every_property_declares_both_endpoints(self) -> None:
        """Property selection hard-gates on domain and range; one missing is one unreachable."""
        profile = json.loads(PACKAGE_PROFILE.read_text(encoding="utf-8"))
        assert [
            p["label"] for p in profile["properties"] if not p.get("domain") or not p.get("range")
        ] == []

    def test_a_bare_name_resolves_to_the_bundled_vocabulary(self) -> None:
        assert GraphKnowsSettings(
            GRAPHKNOWS_RELATION_ONTOLOGY="personal"
        ).relation_ontology_source == str(PACKAGE_PROFILE)

    def test_the_name_is_case_insensitive(self) -> None:
        assert (
            GraphKnowsSettings(GRAPHKNOWS_RELATION_ONTOLOGY="Personal").relation_ontology_source
            == GraphKnowsSettings(GRAPHKNOWS_RELATION_ONTOLOGY="personal").relation_ontology_source
        )

    def test_an_explicit_path_still_wins(self) -> None:
        assert (
            GraphKnowsSettings(
                GRAPHKNOWS_RELATION_ONTOLOGY="/somewhere/else.json"
            ).relation_ontology_source
            == "/somewhere/else.json"
        )

    def test_cco_remains_the_default(self) -> None:
        """Unifying on CCO was a deliberate, documented decision. Reversing it is
        a measurement, not an opinion — so this ticket ships the alternative
        without flipping the default."""
        s = GraphKnowsSettings()
        assert s.relation_ontology_source.endswith("cco.json")
