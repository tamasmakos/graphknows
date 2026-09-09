"""The registry is fixed: core collectors, no ``enable_x``, no pack channel yet."""

from __future__ import annotations

import inspect

from graphknows.channels import registry
from graphknows.channels.base import Channel
from graphknows.channels.registry import core_channels
from graphknows.packs.protocol import DomainPack


def test_the_core_collectors_are_the_same_on_every_call() -> None:
    assert [c.name for c in core_channels()] == ["ontology"]
    assert all(isinstance(c, Channel) for c in core_channels())


def test_the_registry_takes_no_configuration() -> None:
    """No settings argument: a retrieval leg is not an environment switch."""
    assert not inspect.signature(core_channels).parameters


def test_no_enable_flag_survives_in_the_registry() -> None:
    assert "enable_" not in inspect.getsource(registry)


def test_pack_channels_arrive_with_the_dialogue_pack() -> None:
    """FR-022: ``channel()`` is not on the protocol yet, so packs add no legs."""
    assert not hasattr(DomainPack, "channel")
