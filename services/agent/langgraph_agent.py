"""Compatibility shim — import from ``agent.graph.agent`` directly instead."""
from agent.graph.agent import *  # noqa: F401, F403
from agent.graph.agent import (  # noqa: F401 (explicit re-export for IDEs)
    build_graph,
    clear_session_state,
    _session_docs,
    _session_entities,
    _score_query_overlap,
    _normalize_search_eu_sources,
    _choose_search_web_snippet,
)
