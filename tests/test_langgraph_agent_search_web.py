from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

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


def _build_search_web_tool(agent_module):
    _FakeToolNode.last_tools = []
    agent_module.ChatGroq = _FakeChatGroq
    agent_module.ToolNode = _FakeToolNode
    agent_module.StateGraph = _FakeStateGraph
    agent_module.END = object()
    agent_module.build_graph(None, "session-search-web", "fake-key")
    return next(tool for tool in _FakeToolNode.last_tools if tool.__name__ == "search_web")


def test_choose_search_web_snippet_prefers_query_relevant_sentence_over_nav_chrome():
    agent = _load_langgraph_agent_module()

    snippet = agent._choose_search_web_snippet(
        query="Ursula von der Leyen 2024 defence industrial policy European Defence Industrial Strategy EDIS EDIP",
        title="European Defence Industrial Strategy analysis",
        raw_content=(
            "The Institute / About Us / Boards / Experts / Staff / Library / Members login\n"
            "The European Defence Industrial Strategy (EDIS) sets out 2024 defence industrial "
            "policy priorities, links EDIP to capability readiness, and frames the von der Leyen "
            "Commission approach."
        ),
        content="The Institute / About Us / Boards / Experts / Staff / Library / Members login",
    )

    assert "European Defence Industrial Strategy (EDIS)" in snippet
    assert "Members login" not in snippet
    assert "About Us" not in snippet


@pytest.mark.asyncio
async def test_search_web_stores_cleaned_relevant_snippet(monkeypatch):
    agent = _load_langgraph_agent_module()
    search_web = _build_search_web_tool(agent)

    class _FakeAsyncTavilyClient:
        def __init__(self, api_key: str):
            self.api_key = api_key

        async def search(self, **_kwargs):
            return {
                "results": [
                    {
                        "title": "IAI analysis of EDIS and EDIP",
                        "url": "https://example.test/edis",
                        "published_date": "2024-03-05",
                        "content": "The Institute / About Us / Boards / Experts / Staff / Library / Members login",
                        "raw_content": (
                            "The Institute / About Us / Boards / Experts / Staff / Library / Members login\n"
                            "The European Defence Industrial Strategy (EDIS) sets out 2024 defence industrial "
                            "policy priorities, ties EDIP to capability readiness, and explains how Ursula von der "
                            "Leyen's Commission framed the package."
                        ),
                    }
                ]
            }

    fake_tavily_module = types.ModuleType("tavily")
    fake_tavily_module.AsyncTavilyClient = _FakeAsyncTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", fake_tavily_module)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    agent._session_docs.clear()

    result = await search_web(
        query="Ursula von der Leyen 2024 defence industrial policy European Defence Industrial Strategy EDIS EDIP",
        max_results=1,
    )

    stored_docs = agent._session_docs["session-search-web"]
    assert len(stored_docs) == 1
    assert "European Defence Industrial Strategy (EDIS)" in stored_docs[0]["text"]
    assert "Members login" not in stored_docs[0]["text"]
    assert "policy priorities" in result

    os.environ.pop("TAVILY_API_KEY", None)
