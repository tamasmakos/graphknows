from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SERVICES = REPO / "services"
for path in (REPO, SERVICES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from services.search.base_source import BaseSource


def _install_langgraph_agent_import_stubs() -> None:
    if "langgraph.graph" not in sys.modules:
        graph_module = types.ModuleType("langgraph.graph")
        graph_module.StateGraph = type("StateGraph", (), {})
        graph_module.END = object()
        sys.modules["langgraph.graph"] = graph_module

    if "langgraph.graph.message" not in sys.modules:
        graph_message = types.ModuleType("langgraph.graph.message")
        graph_message.add_messages = lambda messages, new_messages: messages + new_messages
        sys.modules["langgraph.graph.message"] = graph_message

    if "langgraph.prebuilt" not in sys.modules:
        prebuilt_module = types.ModuleType("langgraph.prebuilt")
        prebuilt_module.ToolNode = type("ToolNode", (), {})
        sys.modules["langgraph.prebuilt"] = prebuilt_module

    if "langchain_groq" not in sys.modules:
        groq_module = types.ModuleType("langchain_groq")
        groq_module.ChatGroq = type("ChatGroq", (), {})
        sys.modules["langchain_groq"] = groq_module

    if "langchain_core.messages" not in sys.modules:
        messages_module = types.ModuleType("langchain_core.messages")
        messages_module.SystemMessage = type("SystemMessage", (), {})
        messages_module.HumanMessage = type("HumanMessage", (), {})
        sys.modules["langchain_core.messages"] = messages_module

    if "langchain_core.tools" not in sys.modules:
        tools_module = types.ModuleType("langchain_core.tools")
        tools_module.tool = lambda func=None, **_kwargs: func
        sys.modules["langchain_core.tools"] = tools_module


def _load_langgraph_agent_module():
    _install_langgraph_agent_import_stubs()
    sys.modules.pop("services.agent.langgraph_agent", None)
    return importlib.import_module("services.agent.langgraph_agent")


class _FakeChatGroq:
    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, tools, tool_choice=None):
        return self


class _FakeToolNode:
    last_tools = []

    def __init__(self, tools):
        self.__class__.last_tools = list(tools)


class _FakeStateGraph:
    def __init__(self, *_args, **_kwargs):
        pass

    def add_node(self, *_args, **_kwargs):
        return None

    def set_entry_point(self, *_args, **_kwargs):
        return None

    def add_conditional_edges(self, *_args, **_kwargs):
        return None

    def add_edge(self, *_args, **_kwargs):
        return None

    def compile(self):
        return object()


def _build_search_eu_tool(agent_module):
    _FakeToolNode.last_tools = []
    agent_module.ChatGroq = _FakeChatGroq
    agent_module.ToolNode = _FakeToolNode
    agent_module.StateGraph = _FakeStateGraph
    agent_module.END = object()
    agent_module.build_graph(None, "session-test", "fake-key")
    return next(tool for tool in _FakeToolNode.last_tools if tool.__name__ == "search_eu")


class _FakeEPRSBriefingsSource(BaseSource):
    source_id = "eprs_briefings"
    source_type = "EPRS_BRIEFING"
    institution = "EP Research Service"

    search_query_calls = []
    scrape_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def scrape(self, limit: int = 10):
        self.__class__.scrape_calls.append(limit)
        return [
            SimpleNamespace(
                doc_id="doc-weak",
                title="Europe energy outlook",
                raw_text="The report discusses Europe in broad terms.",
                markdown="Weak overlap only.",
                url="https://example.test/doc-weak",
                date="2025-01-15",
            ),
            SimpleNamespace(
                doc_id="doc-older",
                title="Renewable hydrogen industrial strategy policy note",
                raw_text=(
                    "This recent EPRS briefing explains how the EU renewable hydrogen "
                    "clean industrial strategy affects competitiveness, state aid policy, "
                    "and manufacturing support."
                ),
                markdown="Relevant but older.",
                url="https://example.test/doc-older",
                date="2024-02-10",
            ),
            SimpleNamespace(
                doc_id="doc-newer",
                title="Renewable hydrogen industrial strategy policy memo",
                raw_text=(
                    "This recent EPRS briefing explains how the EU renewable hydrogen "
                    "clean industrial strategy affects competitiveness, state aid policy, "
                    "and manufacturing support."
                ),
                markdown="Relevant and newer.",
                url="https://example.test/doc-newer",
                date="2025-03-20",
            ),
        ]

    async def fetch_one(self, url: str):
        raise NotImplementedError

    async def search_query(self, query: str, limit: int = 10):
        self.__class__.search_query_calls.append((query, limit))
        lowered_query = query.lower()
        if "renewable hydrogen" in lowered_query or "clean industrial strategy" in lowered_query:
            return await self.scrape(limit=limit)
        if "edis" in query.lower() or "edip" in query.lower() or "defence readiness" in query.lower():
            return [
                SimpleNamespace(
                    doc_id="doc-edis",
                    title="European defence industrial strategy",
                    raw_text=(
                        "The briefing explains the European defence industrial strategy, "
                        "its readiness goals, and the von der Leyen Commission context."
                    ),
                    markdown="EDIS briefing.",
                    url="https://example.test/doc-edis",
                    date="2024-03-05",
                ),
                SimpleNamespace(
                    doc_id="doc-edip",
                    title="European defence industry programme (EDIP)",
                    raw_text=(
                        "The briefing links EDIP to the European defence industrial strategy "
                        "and defence readiness."
                    ),
                    markdown="EDIP briefing.",
                    url="https://example.test/doc-edip",
                    date="2024-03-05",
                ),
            ]
        return [
            SimpleNamespace(
                doc_id="doc-generic-thinktank",
                title="European defence, strategic autonomy and NATO [What Think Tanks are thinking]",
                raw_text=(
                    "This digest summarizes generic think-tank commentary on European defence, "
                    "strategic autonomy, NATO, and wider policy debates around the Commission."
                ),
                markdown="Generic defence digest.",
                url="https://example.test/doc-generic-thinktank",
                date="2024-04-10",
            ),
            SimpleNamespace(
                doc_id="doc-generic-priorities",
                title="The policy priorities of the von der Leyen II Commission",
                raw_text=(
                    "A broad overview of the Commission priorities, including defence, industrial policy, "
                    "and strategic autonomy themes."
                ),
                markdown="Generic Commission priorities digest.",
                url="https://example.test/doc-generic-priorities",
                date="2024-04-09",
            ),
        ]


class _FakeEPRSRankingRegressionSource(BaseSource):
    source_id = "eprs_briefings"
    source_type = "EPRS_BRIEFING"
    institution = "EP Research Service"

    search_query_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def scrape(self, limit: int = 10):
        raise NotImplementedError

    async def fetch_one(self, url: str):
        raise NotImplementedError

    async def search_query(self, query: str, limit: int = 10):
        self.__class__.search_query_calls.append((query, limit))
        return [
            SimpleNamespace(
                doc_id="doc-white-paper",
                title="White paper on the future of European defence",
                raw_text=(
                    "A broad defence readiness and industrial policy overview covering the "
                    "Commission, defence preparedness, and European defence priorities."
                ),
                markdown="Generic defence white paper.",
                url="https://example.test/doc-white-paper",
                date="2025-03-19",
            ),
            SimpleNamespace(
                doc_id="doc-edis",
                title="European defence industrial strategy",
                raw_text=(
                    "The briefing explains EDIS, defence readiness, and the Commission's "
                    "industrial strategy approach."
                ),
                markdown="EDIS briefing.",
                url="https://example.test/doc-edis",
                date="2024-03-05",
            ),
            SimpleNamespace(
                doc_id="doc-edip",
                title="European defence industry programme (EDIP)",
                raw_text=(
                    "The briefing explains EDIP and its role in the EU defence industry "
                    "programme and strategy."
                ),
                markdown="EDIP briefing.",
                url="https://example.test/doc-edip",
                date="2024-03-05",
            ),
        ]


def test_score_query_overlap_matches_long_natural_language_query():
    agent = _load_langgraph_agent_module()

    query = (
        "Find recent EPRS briefings explaining how the EU renewable hydrogen and "
        "clean industrial strategy affect competitiveness and state aid policy"
    )
    title = "Renewable hydrogen industrial strategy and EU competitiveness"
    text = "This EPRS briefing analyses renewable hydrogen, clean industry, competitiveness, and state aid reform."

    assert query.lower() not in f"{title} {text}".lower()
    assert agent._score_query_overlap(query, title, text) >= 5
    assert agent._score_query_overlap("the and of to", title, text) == 0


def test_score_search_eu_document_boosts_exact_edis_edip_titles():
    agent = _load_langgraph_agent_module()

    query = "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024"
    candidate_queries = [query]

    white_paper_score = agent._score_search_eu_document(
        query,
        candidate_queries,
        "White paper on the future of European defence",
        "A broad defence readiness and industrial policy overview.",
    )
    edis_score = agent._score_search_eu_document(
        query,
        candidate_queries,
        "European defence industrial strategy",
        "The briefing explains EDIS and defence readiness.",
    )
    edip_score = agent._score_search_eu_document(
        query,
        candidate_queries,
        "European defence industry programme (EDIP)",
        "The briefing links EDIP to the EU defence industry programme and strategy.",
    )

    assert edis_score > white_paper_score
    assert edip_score > white_paper_score


def test_normalize_search_eu_sources_replaces_noisy_mix_with_safe_set():
    agent = _load_langgraph_agent_module()

    assert agent._normalize_search_eu_sources("ec_press,eprs_briefings,ep_debates") == [
        "eprs_briefings"
    ]
    assert agent._normalize_search_eu_sources("ep_debates,ec_press") == [
        "eprs_briefings"
    ]


def test_search_eu_candidate_queries_adds_conservative_eprs_expansion():
    agent = _load_langgraph_agent_module()

    assert agent._search_eu_candidate_queries(
        "eprs_briefings",
        "Ursula von der Leyen European Defence Industrial Strategy",
    ) == [
        "Ursula von der Leyen European Defence Industrial Strategy",
        "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024",
    ]
    assert agent._search_eu_candidate_queries(
        "eprs_briefings",
        "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024",
    ) == [
        "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024"
    ]
    assert agent._search_eu_candidate_queries(
        "ec_press",
        "Ursula von der Leyen European Defence Industrial Strategy",
    ) == [
        "Ursula von der Leyen European Defence Industrial Strategy"
    ]


@pytest.mark.asyncio
async def test_search_eu_uses_overlap_ranking_and_eprs_default_source(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_eu = _build_search_eu_tool(agent)

    fake_eprs_module = types.ModuleType("search.adapters.eprs_briefings")
    fake_eprs_module.EPRSBriefingsSource = _FakeEPRSBriefingsSource
    monkeypatch.setitem(sys.modules, "search.adapters.eprs_briefings", fake_eprs_module)

    result = await search_eu(
        query=(
            "Find recent EPRS briefings explaining how the EU renewable hydrogen and "
            "clean industrial strategy affect competitiveness and state aid policy"
        ),
        sources="",
        max_docs=2,
    )

    assert "No EU documents found" not in result
    assert "doc-weak" not in result
    assert "1. Renewable hydrogen industrial strategy policy memo" in result
    assert "2. Renewable hydrogen industrial strategy policy note" in result


@pytest.mark.asyncio
async def test_search_eu_prefers_query_aware_eprs_path(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_eu = _build_search_eu_tool(agent)

    _FakeEPRSBriefingsSource.search_query_calls = []
    _FakeEPRSBriefingsSource.scrape_calls = []

    fake_eprs_module = types.ModuleType("search.adapters.eprs_briefings")
    fake_eprs_module.EPRSBriefingsSource = _FakeEPRSBriefingsSource
    monkeypatch.setitem(sys.modules, "search.adapters.eprs_briefings", fake_eprs_module)

    result = await search_eu(
        query="Ursula von der Leyen European Defence Industrial Strategy",
        sources="eprs_briefings",
        max_docs=2,
    )

    assert _FakeEPRSBriefingsSource.search_query_calls == [
        ("Ursula von der Leyen European Defence Industrial Strategy", 2),
        (
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024",
            2,
        ),
    ]
    assert _FakeEPRSBriefingsSource.scrape_calls == []
    assert "1. European defence industrial strategy" in result
    assert "2. European defence industry programme (EDIP)" in result
    assert "What Think Tanks are thinking" not in result


@pytest.mark.asyncio
async def test_search_eu_gives_query_aware_eprs_path_more_time(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_eu = _build_search_eu_tool(agent)

    _FakeEPRSBriefingsSource.search_query_calls = []

    wait_for_timeouts = []
    original_wait_for = agent.asyncio.wait_for

    async def _record_wait_for(awaitable, timeout):
        wait_for_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(agent.asyncio, "wait_for", _record_wait_for)

    fake_eprs_module = types.ModuleType("search.adapters.eprs_briefings")
    fake_eprs_module.EPRSBriefingsSource = _FakeEPRSBriefingsSource
    monkeypatch.setitem(sys.modules, "search.adapters.eprs_briefings", fake_eprs_module)

    result = await search_eu(
        query=(
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP "
            "defence readiness 2024"
        ),
        sources="eprs_briefings,ec_press_releases,ep_debates",
        max_docs=2,
    )

    assert _FakeEPRSBriefingsSource.search_query_calls == [
        (
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP "
            "defence readiness 2024",
            2,
        )
    ]
    assert agent._SEARCH_EU_QUERY_AWARE_TIMEOUT_SECONDS in wait_for_timeouts
    assert agent._SEARCH_EU_ADAPTER_TIMEOUT_SECONDS not in wait_for_timeouts
    assert "No EU documents found" not in result


@pytest.mark.asyncio
async def test_search_eu_normalizes_noisy_sources_to_eprs(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_eu = _build_search_eu_tool(agent)

    _FakeEPRSBriefingsSource.search_query_calls = []
    _FakeEPRSBriefingsSource.scrape_calls = []

    fake_eprs_module = types.ModuleType("search.adapters.eprs_briefings")
    fake_eprs_module.EPRSBriefingsSource = _FakeEPRSBriefingsSource
    monkeypatch.setitem(sys.modules, "search.adapters.eprs_briefings", fake_eprs_module)

    result = await search_eu(
        query="Ursula von der Leyen European Defence Industrial Strategy",
        sources="ec_press,eprs_briefings,ep_debates",
        max_docs=2,
    )

    assert _FakeEPRSBriefingsSource.search_query_calls == [
        ("Ursula von der Leyen European Defence Industrial Strategy", 2),
        (
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP defence readiness 2024",
            2,
        ),
    ]
    assert _FakeEPRSBriefingsSource.scrape_calls == []
    assert "European defence industrial strategy" in result


@pytest.mark.asyncio
async def test_search_eu_ranks_exact_edis_edip_docs_above_generic_white_paper(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_eu = _build_search_eu_tool(agent)

    _FakeEPRSRankingRegressionSource.search_query_calls = []

    fake_eprs_module = types.ModuleType("search.adapters.eprs_briefings")
    fake_eprs_module.EPRSBriefingsSource = _FakeEPRSRankingRegressionSource
    monkeypatch.setitem(sys.modules, "search.adapters.eprs_briefings", fake_eprs_module)

    result = await search_eu(
        query=(
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP "
            "defence readiness 2024"
        ),
        sources="eprs_briefings",
        max_docs=3,
    )

    assert _FakeEPRSRankingRegressionSource.search_query_calls == [
        (
            "Ursula von der Leyen European Defence Industrial Strategy EDIS EDIP "
            "defence readiness 2024",
            3,
        )
    ]
    assert "1. European defence industrial strategy" in result
    assert "2. European defence industry programme (EDIP)" in result
    assert "3. White paper on the future of European defence" in result
