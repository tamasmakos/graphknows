"""``Memory.build_topics`` and both its surfaces (MCP tool, CLI), plus
``ltm_topics`` / ``ltm_entity``, which inline their own Cypher.

FR-043: none of these had a single test exercising the call path an MCP
client or the CLI actually takes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory import Memory
from graphknows.settings import TopicMode
from graphknows.topics import TopicConfig, TopicSet


def _memory() -> Memory:
    mem = Memory(settings=MagicMock(topic_mode=TopicMode.extractive), namespace="ns")
    mem._get_retriever = AsyncMock()
    mem._store = MagicMock(name="store")
    return mem


class TestMemoryBuildTopics:
    """The orchestration: config resolution, persist gating, export shapes."""

    @staticmethod
    async def _run(
        mem: Memory, topic_set: TopicSet, **kwargs: Any
    ) -> tuple[dict[str, Any], AsyncMock, AsyncMock]:
        with (
            patch("graphknows.topics.build_topics", AsyncMock(return_value=topic_set)) as build,
            patch("graphknows.topics.persist_topics", AsyncMock(return_value=3)) as persist,
        ):
            out = await mem.build_topics(**kwargs)
        return out, build, persist

    @pytest.mark.asyncio
    async def test_default_config_uses_settings_topic_mode(self) -> None:
        mem = _memory()
        out, build, _ = await self._run(mem, TopicSet(metrics={"topics": 1}, namespace="ns"))

        cfg_used = build.await_args.args[1]
        assert cfg_used.mode is TopicMode.extractive
        assert out["persisted"] == 3
        assert out["topics"] == 1

    @pytest.mark.asyncio
    async def test_overrides_replace_fields_on_the_given_config(self) -> None:
        mem = _memory()
        cfg = TopicConfig(min_size=2, mode=TopicMode.extractive)
        out, build, _ = await self._run(mem, TopicSet(namespace="ns"), config=cfg, min_size=9)

        cfg_used = build.await_args.args[1]
        assert cfg_used.min_size == 9
        assert out["config_fingerprint"] == cfg_used.fingerprint()

    @pytest.mark.asyncio
    async def test_persist_false_skips_the_write(self) -> None:
        mem = _memory()
        out, _, persist = await self._run(mem, TopicSet(namespace="ns"), persist=False)

        persist.assert_not_awaited()
        assert out["persisted"] == 0

    @pytest.mark.asyncio
    async def test_include_export_returns_the_payload_inband(self) -> None:
        mem = _memory()
        topic_set = TopicSet(metrics={"topics": 1}, namespace="ns")
        out, _, _ = await self._run(mem, topic_set, include_export=True)

        assert out["export"] == topic_set.to_dict()

    @pytest.mark.asyncio
    async def test_export_path_writes_a_real_json_artifact(self, tmp_path: Any) -> None:
        mem = _memory()
        path = tmp_path / "out.json"
        out, _, _ = await self._run(mem, TopicSet(namespace="ns"), export_path=str(path))

        assert out["export_path"] == str(path)
        assert path.exists()

    @pytest.mark.asyncio
    async def test_connects_the_store_before_building(self) -> None:
        mem = _memory()
        await self._run(mem, TopicSet(namespace="ns"))

        mem._get_retriever.assert_awaited_once()


class _FakeCliMemory:
    """Stand-in for ``Memory`` as constructed by ``graphknows-topics``."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def build_topics(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "topics": 2,
            "mean_size": 3.0,
            "largest_topic": 4,
            "newman_modularity": 0.3,
            "mean_coherence": 0.8,
            "chunk_coverage": 0.5,
            "entity_coverage": 0.6,
            "persisted": 2,
        }

    async def close(self) -> None:
        self.closed = True


class TestCliTopicsSurface:
    """``graphknows-topics``: the CLI surface over ``Memory.build_topics``."""

    @pytest.mark.asyncio
    async def test_single_run_persists_by_default_and_prints_metrics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from graphknows.cli import topics as cli_topics

        fake = _FakeCliMemory()
        with patch("graphknows.memory.Memory", return_value=fake):
            code = await cli_topics.amain(["--mode", "extractive"])

        assert code == 0
        assert fake.calls[0]["persist"] is True
        assert fake.closed is True
        assert "newman_modularity" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_no_persist_flag_is_forwarded(self) -> None:
        from graphknows.cli import topics as cli_topics

        fake = _FakeCliMemory()
        with patch("graphknows.memory.Memory", return_value=fake):
            await cli_topics.amain(["--no-persist"])

        assert fake.calls[0]["persist"] is False

    @pytest.mark.asyncio
    async def test_export_path_is_forwarded(self, tmp_path: Any) -> None:
        from graphknows.cli import topics as cli_topics

        fake = _FakeCliMemory()
        export_path = str(tmp_path / "out.json")
        with patch("graphknows.memory.Memory", return_value=fake):
            await cli_topics.amain(["--export", export_path])

        assert fake.calls[0]["export_path"] == export_path

    @pytest.mark.asyncio
    async def test_sweep_calls_build_topics_per_value_never_persisting(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from graphknows.cli import topics as cli_topics

        fake = _FakeCliMemory()
        with patch("graphknows.memory.Memory", return_value=fake):
            code = await cli_topics.amain(["--sweep", "2,3,5"])

        assert code == 0
        assert [c["config"].min_size for c in fake.calls] == [2, 3, 5]
        assert all(c["persist"] is False for c in fake.calls)
        assert "newman Q" in capsys.readouterr().out


class TestMcpTopicsSurface:
    """``memory_topics``: config resolution, then a thin delegate to the runtime."""

    @pytest.mark.asyncio
    async def test_empty_mode_falls_back_to_settings_and_clamps_min_size(self) -> None:
        from graphknows.server.mcp.tools.topics import memory_topics

        runtime = MagicMock()
        runtime.settings = MagicMock(topic_mode=TopicMode.llm)
        runtime.build_topics = AsyncMock(return_value={"topics": 1})

        with patch(
            "graphknows.server.mcp.tools.topics.get_runtime", AsyncMock(return_value=runtime)
        ):
            out = await memory_topics(mode="", min_size=0)

        cfg = runtime.build_topics.await_args.kwargs["config"]
        assert cfg.mode is TopicMode.llm
        assert cfg.min_size == 1  # max(1, 0)
        assert out == {"topics": 1}

    @pytest.mark.asyncio
    async def test_explicit_mode_and_persist_flow_through(self) -> None:
        from graphknows.server.mcp.tools.topics import memory_topics

        runtime = MagicMock()
        runtime.settings = MagicMock(topic_mode=TopicMode.llm)
        runtime.build_topics = AsyncMock(return_value={"topics": 3})

        with patch(
            "graphknows.server.mcp.tools.topics.get_runtime", AsyncMock(return_value=runtime)
        ):
            await memory_topics(mode="extractive", min_size=5, persist=False)

        kwargs = runtime.build_topics.await_args.kwargs
        assert kwargs["config"].mode is TopicMode.extractive
        assert kwargs["config"].min_size == 5
        assert kwargs["persist"] is False
        assert kwargs["include_export"] is True


class TestLtmToolsDelegate:
    """``ltm_topics`` / ``ltm_entity`` MCP tools: thin delegates to the runtime."""

    @pytest.mark.asyncio
    async def test_ltm_topics_tool_delegates(self) -> None:
        from graphknows.server.mcp.tools.ltm import ltm_topics

        runtime = MagicMock()
        runtime.ltm_topics = AsyncMock(return_value={"topics": []})

        with patch("graphknows.server.mcp.tools.ltm.get_runtime", AsyncMock(return_value=runtime)):
            out = await ltm_topics(min_entities=5)

        runtime.ltm_topics.assert_awaited_once_with(min_entities=5)
        assert out == {"topics": []}

    @pytest.mark.asyncio
    async def test_ltm_entity_tool_delegates(self) -> None:
        from graphknows.server.mcp.tools.ltm import ltm_entity

        runtime = MagicMock()
        runtime.ltm_entity = AsyncMock(return_value={"name": "Acme"})

        with patch("graphknows.server.mcp.tools.ltm.get_runtime", AsyncMock(return_value=runtime)):
            out = await ltm_entity(name="Acme")

        runtime.ltm_entity.assert_awaited_once_with(name="Acme")
        assert out == {"name": "Acme"}


class _FakeStore:
    """Stand-in for the ArcadeDB store, recording the Cypher it was handed."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.query_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def connect(self) -> None:
        pass

    async def query(self, cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append((cypher, kwargs))
        return self._rows

    async def close(self) -> None:
        self.closed = True


class TestMemoryLtmTopics:
    """``Memory.ltm_topics``: the inline Cypher's own bind params and shaping."""

    @pytest.mark.asyncio
    async def test_passes_min_entities_and_shapes_the_result(self) -> None:
        rows = [
            {
                "@rid": "#1:1",
                "topic_id": "t1",
                "title": "Trips",
                "summary": "s",
                "keywords": ["k"],
                "coherence": 0.9,
                "entity_count": 25,
                "entity_names": [f"e{i}" for i in range(25)],
            }
        ]
        store = _FakeStore(rows)
        mem = Memory(settings=MagicMock(), namespace="ns")

        with patch("graphknows.storage.build_graph_store", return_value=store):
            out = await mem.ltm_topics(min_entities=3)

        assert store.query_calls[0][1] == {"min": 3}
        topic = out["topics"][0]
        assert "@rid" not in topic
        assert len(topic["entity_names"]) == 20  # ENTITY.name is truncated, not the count
        assert topic["entity_count"] == 25
        assert store.closed is True


class TestMemoryLtmEntity:
    """``Memory.ltm_entity``: found vs. not-found shapes of the inline Cypher."""

    @pytest.mark.asyncio
    async def test_found_entity_strips_the_rid(self) -> None:
        rows = [
            {
                "@rid": "#1:2",
                "name": "Acme",
                "type": "ORG",
                "confidence": 0.9,
                "pagerank": 0.1,
                "community_id": 2,
            }
        ]
        store = _FakeStore(rows)
        mem = Memory(settings=MagicMock(), namespace="ns")

        with patch("graphknows.storage.build_graph_store", return_value=store):
            out = await mem.ltm_entity("acme")

        assert store.query_calls[0][1] == {"name": "acme"}
        assert out == {
            "name": "Acme",
            "type": "ORG",
            "confidence": 0.9,
            "pagerank": 0.1,
            "community_id": 2,
        }

    @pytest.mark.asyncio
    async def test_missing_entity_reports_not_found(self) -> None:
        store = _FakeStore([])
        mem = Memory(settings=MagicMock(), namespace="ns")

        with patch("graphknows.storage.build_graph_store", return_value=store):
            out = await mem.ltm_entity("ghost")

        assert out == {"error": "Entity not found: ghost"}
