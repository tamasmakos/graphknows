"""Configuring a GraphKnowsMemory is one argument, and a conflict is an error.

``GraphKnowsSettings`` is 26 knobs deep with defaults that make ~2 of them
necessary, but none of it was reachable from this seam: a caller who needed a
different embed model or ``llm_assisted`` had to build a ``Memory`` by hand and
pass ``memory=`` — at which point ``namespace=`` was silently dropped.
"""

from __future__ import annotations

import pytest

from graphknows.exceptions import ConfigurationError
from graphknows.integrations.langgraph import GraphKnowsMemory
from graphknows.settings import GraphKnowsSettings
from tests.fixtures.memory import FakeMemory as _FakeMemory


def test_settings_reach_the_memory_it_builds() -> None:
    settings = GraphKnowsSettings(embed_model="a-different-model")

    gm = GraphKnowsMemory("s1", settings=settings, namespace="demo")

    assert gm.memory.settings.embed_model == "a-different-model"
    assert gm.memory.namespace == "demo"


def test_channels_reach_the_memory_it_builds() -> None:
    """The extension seam has to be reachable from the integration, not only
    from ``Memory`` — an agent framework is exactly the caller that wants it."""
    from graphknows.channels.base import Channel

    channel = Channel()

    gm = GraphKnowsMemory("s1", namespace="demo", channels=[channel])

    assert gm.memory._channels == [channel]


@pytest.mark.parametrize("conflict", [{"namespace": "demo"}, {"settings": None}, {"channels": []}])
def test_configuring_a_supplied_memory_is_refused(conflict: dict) -> None:
    """Silently dropping namespace= sent a caller's turns to the wrong database.

    The failure surfaced three steps later as "memory is empty", with nothing
    at the call site to suggest the namespace had been ignored.
    """
    if "settings" in conflict:
        conflict["settings"] = GraphKnowsSettings()

    with pytest.raises(ConfigurationError, match="not both"):
        GraphKnowsMemory("s1", memory=_FakeMemory(), **conflict)
