"""graphknows service entrypoint — supports HTTP (default), stdio, and SSE transports."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    """Parse CLI args and start the graphknows service."""
    parser = argparse.ArgumentParser(description="graphknows service")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio", "sse"],
        default="http",
        help=(
            "MCP transport mode. "
            "'http' (default): FastAPI + FastMCP over HTTP on port 8025. "
            "'stdio': FastMCP only over stdin/stdout — for Claude Desktop / local Cursor. "
            "'sse': FastAPI + FastMCP SSE sub-app — for streaming MCP clients."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8025,
        help="HTTP port (http and sse modes only).",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdio: bypass FastAPI, run MCP server over stdin/stdout.
        # Stores are initialised inline before entering the MCP event loop.
        _run_stdio()
    elif args.transport == "sse":
        # SSE: mount FastMCP SSE sub-app alongside the FastAPI app.
        _run_sse(port=args.port)
    else:
        # Default HTTP mode: FastAPI + FastMCP streamable-HTTP sub-app.
        uvicorn.run(  # NOSONAR — containerised; network boundary enforced by Docker/k8s
            "graphknows.api:app",
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            access_log=True,
        )


def _run_stdio() -> None:
    """Run the MCP server in stdio transport mode.

    Initialises stores synchronously then hands off to FastMCP's stdio loop.
    Suitable for Claude Desktop and local Cursor MCP integration.
    """
    import asyncio
    import logging

    from graphknows.config import GraphKnowsSettings
    from graphknows.mcp.server import mcp, set_dependencies
    from graphknows.retrieval.retriever import HybridRetriever
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.neo4j import Neo4jStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    logging.basicConfig(level=logging.WARNING)  # keep stdio clean

    async def _init() -> None:
        cfg = GraphKnowsSettings()
        neo = Neo4jStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password, cfg.neo4j_database)
        pg = PgVectorStore(dsn=cfg.pgvector_dsn)
        mg = MemgraphStore(uri=cfg.memgraph_uri)
        await pg.connect()
        await asyncio.gather(pg.ensure_schema(), neo.ensure_indexes())
        retriever = HybridRetriever(pg=pg, neo=neo, embed_model=cfg.embed_model)
        set_dependencies(retriever, pg, neo, mg, {"embed_model": cfg.embed_model})

    asyncio.run(_init())
    mcp.run(transport="stdio")


def _run_sse(port: int = 8025) -> None:
    """Run FastAPI + FastMCP SSE sub-app for streaming MCP clients."""
    from graphknows.api import app
    from graphknows.mcp.server import mcp

    # Mount SSE endpoint alongside the existing HTTP app
    app.mount("/mcp-sse", mcp.sse_app())
    uvicorn.run(  # NOSONAR — containerised; network boundary enforced by Docker/k8s
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
