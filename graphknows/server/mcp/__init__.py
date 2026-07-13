"""MCP sub-package: FastMCP server for graphknows tools."""
# NOTE: do NOT import from graphknows.server.mcp.stdio_server here — stdio_server.py
# imports from the third-party `mcp` package, which Python resolves as this
# package during initialization, causing a circular import.
# Consumers should import directly: from graphknows.server.mcp.stdio_server import app
