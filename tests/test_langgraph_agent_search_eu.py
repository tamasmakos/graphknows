from __future__ import annotations

import importlib
import sys
import types
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SERVICES = REPO / "services"
for path in (REPO, SERVICES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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
    for key in (
        "langgraph.prebuilt",
        "langchain_core.tools",
        "langchain_groq",
        "langchain_core.messages",
    ):
        sys.modules.pop(key, None)
    _install_langgraph_agent_import_stubs()
    sys.modules.pop("services.agent.langgraph_agent", None)
    sys.modules.pop("agent.langgraph_agent", None)
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


class _FakeSearchServiceClient:
    fetch_calls: list[tuple[str, list[str], int]] = []
    response_docs: list[SimpleNamespace] = []
    response_error: Exception | None = None

    async def fetch_documents(self, query: str, sources: list[str], limit: int):
        self.__class__.fetch_calls.append((query, list(sources), limit))
        if self.__class__.response_error is not None:
            raise self.__class__.response_error
        return list(self.__class__.response_docs)


def _build_search_eu_tool(agent_module):
    _FakeToolNode.last_tools = []
    agent_module.ChatGroq = _FakeChatGroq
    agent_module.StateGraph = _FakeStateGraph
    agent_module.END = object()
    # Also patch the canonical module since the shim delegates to agent.graph.agent
    import agent.graph.agent as _canonical_agent
    with (
        unittest.mock.patch.object(agent_module, "ToolNode", _FakeToolNode),
        unittest.mock.patch.object(_canonical_agent, "ChatGroq", _FakeChatGroq),
        unittest.mock.patch.object(_canonical_agent, "StateGraph", _FakeStateGraph),
        unittest.mock.patch.object(_canonical_agent, "ToolNode", _FakeToolNode),
    ):
        agent_module.build_graph(None, "session-test", "fake-key")
    return next(tool for tool in _FakeToolNode.last_tools if tool.__name__ == "search_eu")


def _patch_search_client(
    monkeypatch,
    docs: list[SimpleNamespace] | None = None,
    error: Exception | None = None,
):
    import agent.tools.search.client as search_client_module

    _FakeSearchServiceClient.fetch_calls = []
    _FakeSearchServiceClient.response_docs = list(docs or [])
    _FakeSearchServiceClient.response_error = error
    monkeypatch.setattr(search_client_module, "SearchServiceClient", _FakeSearchServiceClient)


def _raw_doc(
    doc_id: str,
    title: str,
    *,
    text: str = "Document text",
    url: str = "https://example.test/doc",
    source: str = "eprs_briefings",
    date: str = "2025-03-20",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=doc_id,
        title=title,
        text=text,
        url=url,
        source=source,
        date=date,
    )


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


def test_normalize_search_eu_sources_defaults_and_filters_to_safe_sources():
    agent = _load_langgraph_agent_module()

    assert agent._normalize_search_eu_sources("") == [
        "eprs_briefings",
        "ep_resolutions",
        "ep_debates",
    ]
    assert agent._normalize_search_eu_sources("ec_press,eprs_briefings,ep_debates") == [
        "eprs_briefings",
        "ep_debates",
    ]
    assert agent._normalize_search_eu_sources("ec_press,unknown") == [
        "eprs_briefings",
        "ep_resolutions",
        "ep_debates",
    ]


@pytest.mark.asyncio
async def test_search_eu_uses_http_client_with_default_safe_sources(monkeypatch):
    agent = _load_langgraph_agent_module()
    _patch_search_client(
        monkeypatch,
        docs=[
            _raw_doc(
                "doc-1",
                "Renewable hydrogen industrial strategy policy memo",
                text="Relevant excerpt.",
                url="https://example.test/doc-1",
            ),
            _raw_doc(
                "doc-2",
                "Renewable hydrogen industrial strategy policy note",
                text="Second relevant excerpt.",
                url="https://example.test/doc-2",
                date="2024-02-10",
            ),
        ],
    )
    search_eu = _build_search_eu_tool(agent)

    result = await search_eu(
        query=(
            "Find recent EPRS briefings explaining how the EU renewable hydrogen and "
            "clean industrial strategy affect competitiveness and state aid policy"
        ),
        sources="",
        max_docs=2,
    )

    assert _FakeSearchServiceClient.fetch_calls == [
        (
            "Find recent EPRS briefings explaining how the EU renewable hydrogen and clean industrial strategy affect competitiveness and state aid policy",
            ["eprs_briefings", "ep_resolutions", "ep_debates"],
            2,
        )
    ]
    assert "No EU documents found" not in result
    assert "1. Renewable hydrogen industrial strategy policy memo" in result
    assert "2. Renewable hydrogen industrial strategy policy note" in result
    assert "Institution: eprs_briefings" in result


@pytest.mark.asyncio
async def test_search_eu_preserves_requested_safe_source_order(monkeypatch):
    agent = _load_langgraph_agent_module()
    _patch_search_client(
        monkeypatch,
        docs=[
            _raw_doc(
                "doc-1",
                "European defence industrial strategy",
                source="ep_debates",
            )
        ],
    )
    search_eu = _build_search_eu_tool(agent)

    result = await search_eu(
        query="Ursula von der Leyen European Defence Industrial Strategy",
        sources="ep_debates,ec_press,eprs_briefings",
        max_docs=2,
    )

    assert _FakeSearchServiceClient.fetch_calls == [
        (
            "Ursula von der Leyen European Defence Industrial Strategy",
            ["ep_debates", "eprs_briefings"],
            2,
        )
    ]
    assert "1. European defence industrial strategy" in result
    assert "Institution: ep_debates" in result


@pytest.mark.asyncio
async def test_search_eu_returns_empty_message_when_client_fails(monkeypatch):
    agent = _load_langgraph_agent_module()
    _patch_search_client(monkeypatch, error=RuntimeError("boom"))
    search_eu = _build_search_eu_tool(agent)

    result = await search_eu(
        query="Ursula von der Leyen European Defence Industrial Strategy",
        sources="eprs_briefings",
        max_docs=2,
    )

    assert _FakeSearchServiceClient.fetch_calls == [
        (
            "Ursula von der Leyen European Defence Industrial Strategy",
            ["eprs_briefings"],
            2,
        )
    ]
    assert result == "No EU documents found for this query."


@pytest.mark.asyncio
async def test_search_eu_limits_formatted_output_to_max_docs(monkeypatch):
    agent = _load_langgraph_agent_module()
    _patch_search_client(
        monkeypatch,
        docs=[
            _raw_doc("doc-1", "Doc One"),
            _raw_doc("doc-2", "Doc Two"),
            _raw_doc("doc-3", "Doc Three"),
        ],
    )
    search_eu = _build_search_eu_tool(agent)

    result = await search_eu(
        query="industrial policy",
        sources="eprs_briefings",
        max_docs=2,
    )

    assert "1. Doc One" in result
    assert "2. Doc Two" in result
    assert "3. Doc Three" not in result