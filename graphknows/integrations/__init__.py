"""integrations — adapters for agent frameworks and the out-of-process client SDK.

One subpackage per target, each owning its own private modules and exposing a
flat public surface from its ``__init__``:

- :mod:`graphknows.integrations.langgraph` — LangGraph hooks + store.
- :mod:`graphknows.integrations.client` — stdlib-only MCP client.

Subpackages import their target framework lazily, so importing this package —
or any adapter's ``__init__`` — pulls no third-party framework.
"""
