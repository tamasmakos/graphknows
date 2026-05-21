"""MCP sub-package: FastMCP server for graphknows tools."""
# NOTE: do NOT import from graphknows.mcp.server here — server.py imports
# from the third-party `mcp` package, which Python resolves as this package
# during initialization, causing a circular import.
# Consumers should import directly: from graphknows.mcp.server import mcp

