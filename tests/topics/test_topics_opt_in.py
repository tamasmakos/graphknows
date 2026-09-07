"""Topic building at flush is opt-in, not a preset side effect (FR-045).

Topics keep exactly one reader — the browse tool, ``ltm_topics`` — so a flush
paying for clustering + description merely because ``GRAPHKNOWS_MODE=
llm_assisted`` prefers ``TopicMode.llm`` as a preset was a cost nobody asked
for. ``Memory._flush_topics`` now acts on the raw ``GRAPHKNOWS_TOPICS``
setting, not the preset-resolved ``topic_mode`` — an explicit rebuild through
``Memory.build_topics`` (and the ``memory_topics`` tool built on it) is
unaffected: calling either one is itself the opt-in, so it keeps following
the preset when the caller doesn't override it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings, MemoryMode, TopicMode
from graphknows.topics import TopicSet


def _settings(**kwargs: Any) -> GraphKnowsSettings:
    """Settings read from nothing but these arguments — no ambient .env."""
    return GraphKnowsSettings(_env_file=None, **kwargs)


def _memory(**settings_kwargs: Any) -> Memory:
    return Memory(settings=_settings(**settings_kwargs), namespace="ns")


@pytest.fixture(autouse=True)
def _no_ambient_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own key or mode in the environment must not steer these."""
    for var in ("GRAPHKNOWS_MODE", "GRAPHKNOWS_TOPICS", "GRAPHKNOWS_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestFlushTimeIsOptIn:
    @pytest.mark.asyncio
    async def test_llm_assisted_preset_alone_does_not_build_at_flush(self) -> None:
        """The preset resolves ``topic_mode`` to ``llm``, but a flush must not
        act on the preset alone."""
        mem = _memory(GRAPHKNOWS_MODE=MemoryMode.llm_assisted, GRAPHKNOWS_LLM_API_KEY="k")
        assert mem.settings.topic_mode is TopicMode.llm  # preset unchanged

        with patch("graphknows.topics.build_topics", AsyncMock()) as build:
            out = await mem._flush_topics(MagicMock(name="store"))

        build.assert_not_awaited()
        assert out == {"topics_created": 0, "modularity": None}

    @pytest.mark.asyncio
    async def test_explicit_topics_setting_opts_a_flush_in(self) -> None:
        mem = _memory(GRAPHKNOWS_MODE=MemoryMode.llm_free, GRAPHKNOWS_TOPICS=TopicMode.extractive)
        topic_set = TopicSet(metrics={"newman_modularity": 0.4}, namespace="ns")

        with (
            patch("graphknows.topics.build_topics", AsyncMock(return_value=topic_set)) as build,
            patch("graphknows.topics.persist_topics", AsyncMock(return_value=5)) as persist,
        ):
            out = await mem._flush_topics(MagicMock(name="store"))

        build.assert_awaited_once()
        persist.assert_awaited_once()
        assert out == {"topics_created": 5, "modularity": 0.4}

    @pytest.mark.asyncio
    async def test_explicit_none_still_skips(self) -> None:
        mem = _memory(
            GRAPHKNOWS_MODE=MemoryMode.llm_assisted,
            GRAPHKNOWS_TOPICS=TopicMode.none,
            GRAPHKNOWS_LLM_API_KEY="k",
        )

        with patch("graphknows.topics.build_topics", AsyncMock()) as build:
            out = await mem._flush_topics(MagicMock(name="store"))

        build.assert_not_awaited()
        assert out == {"topics_created": 0, "modularity": None}


class TestExplicitRebuildIsUnaffected:
    @pytest.mark.asyncio
    async def test_build_topics_still_honours_the_mode_preset(self) -> None:
        """Calling ``Memory.build_topics`` at all is the opt-in act, so it
        keeps resolving the preset when the caller doesn't override it."""
        mem = _memory(GRAPHKNOWS_MODE=MemoryMode.llm_assisted, GRAPHKNOWS_LLM_API_KEY="k")
        mem._get_retriever = AsyncMock()
        mem._store = MagicMock(name="store")
        topic_set = TopicSet(namespace="ns")

        with (
            patch("graphknows.topics.build_topics", AsyncMock(return_value=topic_set)) as build,
            patch("graphknows.topics.persist_topics", AsyncMock(return_value=0)),
        ):
            await mem.build_topics()

        cfg_used = build.await_args.args[1]
        assert cfg_used.mode is TopicMode.llm
