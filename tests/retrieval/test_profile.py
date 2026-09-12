"""The core retrieval profile: its defaults, and that a pack cannot mutate one."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphknows.retrieval.profile import CORE_PROFILE, RetrievalProfile


def test_core_defaults_match_the_current_read_path() -> None:
    """The default profile is what recall already runs on today."""
    assert RetrievalProfile() == CORE_PROFILE
    assert CORE_PROFILE.pool_factor == 4
    assert CORE_PROFILE.min_pool == 100
    assert CORE_PROFILE.neighbor_radius == 0
    assert CORE_PROFILE.date_gate is True
    assert CORE_PROFILE.frame_alpha == 0.5
    assert CORE_PROFILE.channels == frozenset()


def test_a_pack_overrides_only_what_it_names() -> None:
    """Fields a profile leaves out keep their core default."""
    profile = RetrievalProfile(neighbor_radius=2, channels=frozenset({"dialogue"}))

    assert profile.neighbor_radius == 2
    assert profile.channels == frozenset({"dialogue"})
    assert profile.pool_factor == CORE_PROFILE.pool_factor


def test_profile_is_a_frozen_value() -> None:
    """Two packs may hold the same profile, so nobody may write through it."""
    with pytest.raises(ValidationError):
        CORE_PROFILE.neighbor_radius = 3


@pytest.mark.parametrize(
    "field, value",
    [("pool_factor", 0), ("min_pool", -1), ("neighbor_radius", -1), ("frame_alpha", -0.1)],
)
def test_out_of_range_knobs_are_rejected(field: str, value: float) -> None:
    """A nonsensical knob fails at construction, not mid-recall."""
    with pytest.raises(ValidationError):
        RetrievalProfile(**{field: value})
