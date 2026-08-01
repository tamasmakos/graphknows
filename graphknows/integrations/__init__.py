"""integrations — adapters for agent frameworks and the out-of-process client SDK.

Submodules import their target framework lazily, so importing this package pulls
no third-party framework. Use:

- :mod:`graphknows.integrations.langgraph` — LangGraph hooks + store.
- :mod:`graphknows.integrations.client` — stdlib-only ``RemoteMemory`` over MCP.
"""
