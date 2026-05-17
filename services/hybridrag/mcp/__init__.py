"""HybridRAG MCP sub-package.

Imports are deferred to avoid a circular-import between this __init__ and
hybridrag.mcp.server at module load time.  Callers should import directly::

    from hybridrag.mcp.server import mcp, set_dependencies
"""
# Do NOT re-export here — import directly from hybridrag.mcp.server.
__all__ = ["mcp", "set_dependencies"]


def __getattr__(name: str):  # lazy attribute access keeps the namespace clean
    if name in ("mcp", "set_dependencies"):
        from hybridrag.mcp import server as _server  # noqa: PLC0415
        return getattr(_server, name)
    raise AttributeError(name)
