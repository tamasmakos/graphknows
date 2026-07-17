"""Tiny local web server for exploring graphknows memory graphs."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on core-only installs
    from graphknows.exceptions import MissingExtraError

    raise MissingExtraError("The Memory Mirror web server", "web") from exc

from graphknows.settings import GraphKnowsSettings
from graphknows.storage import (
    build_arcadedb_client,
    build_arcadedb_ltm_store,
    build_arcadedb_stm_store,
)
from graphknows.links import DEFAULT_PORT
from graphknows.storage.namespace import db_names, namespaces_from_databases
from graphknows.memory import Memory

_settings: GraphKnowsSettings | None = None
# One Memory per namespace — the web server browses many tenants in one process.
_runtimes: dict[str, Memory] = {}


def get_settings() -> GraphKnowsSettings:
    """Return web-local settings for the process-level Memory Mirror server."""
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def get_runtime(namespace: str = "") -> Memory:
    """Return the Memory for *namespace*, creating and caching it lazily."""
    if namespace not in _runtimes:
        _runtimes[namespace] = Memory(get_settings(), namespace=namespace)
    return _runtimes[namespace]


def _frontend_dir() -> Path:
    configured = os.environ.get("GRAPHKNOWS_FRONTEND_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    source_tree = Path(__file__).resolve().parents[2] / "frontend"
    if source_tree.exists():
        return source_tree
    return Path.cwd() / "frontend"


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {"error": exc.__class__.__name__, "message": str(exc)}


def _string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _number(value: Any, fallback: float = 1.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _row_value(row: Any, key: str) -> Any:
    if hasattr(row, "get"):
        return row.get(key)
    return row[key]


def _normalise_stm_graph(raw: dict[str, Any], limit: int) -> dict[str, Any]:
    nodes = []
    known_ids: set[str] = set()
    for item in raw.get("nodes", [])[:limit]:
        node_id = _string(item.get("id") or item.get("name") or item.get("label"))
        if not node_id or node_id in known_ids:
            continue
        known_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": _string(item.get("name") or item.get("label") or node_id),
                "type": _string(item.get("type") or item.get("label"), "Entity"),
                "score": _number(item.get("confidence"), 1.0),
                "scope": "stm",
            }
        )

    links = []
    for item in raw.get("edges", []):
        source = _string(item.get("source") or item.get("from"))
        target = _string(item.get("target") or item.get("to"))
        if source in known_ids and target in known_ids:
            links.append(
                {
                    "source": source,
                    "target": target,
                    "label": _string(item.get("label") or item.get("rel_type") or item.get("type")),
                    "weight": _number(item.get("weight"), 1.0),
                }
            )

    return {
        "scope": "stm",
        "session_id": raw.get("session_id", ""),
        "nodes": nodes,
        "links": links,
        "node_count": len(nodes),
        "edge_count": len(links),
    }


def _ltm_queries(session_id: str, limit: int) -> tuple[str, str, dict[str, Any]]:
    if session_id:
        node_cypher = """
        MATCH (:SESSION {id: $session_id})-[:IN_SESSION*1..3]->(x)-[:HAS_ENTITY|MENTIONS]-(e:ENTITY)
        OPTIONAL MATCH (e)-[:IN_TOPIC]->(t:TOPIC)
        WITH DISTINCT e, t
        RETURN coalesce(e.id, e.name) AS id,
               coalesce(e.name, e.id) AS label,
               coalesce(e.label, e.type, 'Entity') AS type,
               coalesce(t.community_id, e.community_id) AS topic,
               t.summary AS topic_summary,
               e.pagerank AS pagerank,
               e.betweenness AS betweenness
        ORDER BY coalesce(e.pagerank, 0.0) DESC, label
        LIMIT $limit
        """
        edge_cypher = """
        MATCH (:SESSION {id: $session_id})-[:IN_SESSION*1..3]->(x)-[:HAS_ENTITY|MENTIONS]-(a:ENTITY)
        MATCH (:SESSION {id: $session_id})-[:IN_SESSION*1..3]->(y)-[:HAS_ENTITY|MENTIONS]-(b:ENTITY)
        MATCH (a)-[r:REL]->(b)
        WHERE a <> b
        WITH DISTINCT a, b, r LIMIT $limit
        RETURN coalesce(a.id, a.name) AS source,
               coalesce(b.id, b.name) AS target,
               coalesce(r.label, r.predicate) AS label,
               coalesce(r.weight, 1) AS weight
        """
        return node_cypher, edge_cypher, {"session_id": session_id, "limit": limit}

    node_cypher = """
    MATCH (e:ENTITY)
    OPTIONAL MATCH (e)-[:IN_TOPIC]->(t:TOPIC)
    RETURN coalesce(e.id, e.name) AS id,
           coalesce(e.name, e.id) AS label,
           coalesce(e.label, e.type, 'Entity') AS type,
           coalesce(t.community_id, e.community_id) AS topic,
           t.summary AS topic_summary,
           e.pagerank AS pagerank,
           e.betweenness AS betweenness
    ORDER BY coalesce(e.pagerank, 0.0) DESC, label
    LIMIT $limit
    """
    edge_cypher = """
    MATCH (a:ENTITY)-[r:REL]->(b:ENTITY)
    WHERE a <> b
    WITH a, b, r.predicate AS label, coalesce(r.weight, 1) AS weight LIMIT $limit
    RETURN coalesce(a.id, a.name) AS source,
           coalesce(b.id, b.name) AS target,
           label,
           weight
    """
    return node_cypher, edge_cypher, {"limit": limit}


async def _ltm_graph(session_id: str, limit: int, namespace: str = "") -> dict[str, Any]:
    ltm = build_arcadedb_ltm_store(get_settings(), namespace)
    await ltm.connect()
    try:
        node_cypher, edge_cypher, params = _ltm_queries(session_id, limit)
        node_rows, edge_rows = await asyncio.gather(
            ltm._qry(node_cypher, **params),
            ltm._qry(edge_cypher, **params),
        )
    finally:
        await ltm.close()

    nodes = []
    known_ids: set[str] = set()
    for row in node_rows:
        node_id = _string(_row_value(row, "id"))
        if not node_id or node_id in known_ids:
            continue
        known_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": _string(_row_value(row, "label") or node_id),
                "type": _string(_row_value(row, "type"), "Entity"),
                "topic": _string(_row_value(row, "topic")),
                "topic_summary": _string(_row_value(row, "topic_summary")),
                "pagerank": _number(_row_value(row, "pagerank"), 1.0),
                "betweenness": _number(_row_value(row, "betweenness"), 0.0),
                "scope": "ltm",
            }
        )

    links = []
    for row in edge_rows:
        source = _string(_row_value(row, "source"))
        target = _string(_row_value(row, "target"))
        if source in known_ids and target in known_ids and source != target:
            links.append(
                {
                    "source": source,
                    "target": target,
                    "label": _string(_row_value(row, "label")),
                    "weight": _number(_row_value(row, "weight"), 1.0),
                }
            )

    return {
        "scope": "ltm",
        "session_id": session_id,
        "nodes": nodes,
        "links": links,
        "node_count": len(nodes),
        "edge_count": len(links),
    }


async def _db_count(
    client: Any, db: str, cypher: str, params: dict[str, Any] | None = None
) -> int | None:
    """Count rows for a query, returning None when the database is absent/unready."""
    try:
        rows = await client.query(db, cypher, params=params)
        return int(rows[0].get("c") or 0) if rows else 0
    except Exception:
        return None


async def _namespace_stats(client: Any, namespace: str, session_id: str = "") -> dict[str, Any]:
    """Return STM + LTM node counts for one namespace (optionally scoped to a session)."""
    stm_db, ltm_db = db_names(namespace)
    node_filter = " {session_id: $sid}" if session_id else ""
    scoped = {"sid": session_id} if session_id else None
    (
        stm_chunks,
        stm_turns,
        stm_entities,
        stm_sessions,
        ltm_chunks,
        ltm_turns,
        ltm_entities,
        ltm_topics,
        ltm_sessions,
    ) = await asyncio.gather(
        _db_count(client, stm_db, f"MATCH (n:CHUNK{node_filter}) RETURN count(n) AS c", scoped),
        _db_count(client, stm_db, f"MATCH (n:TURN{node_filter}) RETURN count(n) AS c", scoped),
        _db_count(client, stm_db, "MATCH (n:ENTITY) RETURN count(n) AS c"),
        _db_count(client, stm_db, "MATCH (n:SESSION) RETURN count(n) AS c"),
        _db_count(client, ltm_db, f"MATCH (n:CHUNK{node_filter}) RETURN count(n) AS c", scoped),
        _db_count(client, ltm_db, f"MATCH (n:TURN{node_filter}) RETURN count(n) AS c", scoped),
        _db_count(client, ltm_db, "MATCH (n:ENTITY) RETURN count(n) AS c"),
        _db_count(client, ltm_db, "MATCH (n:TOPIC) RETURN count(n) AS c"),
        _db_count(client, ltm_db, "MATCH (n:SESSION) RETURN count(n) AS c"),
    )
    return {
        "namespace": namespace,
        "stm_db": stm_db,
        "ltm_db": ltm_db,
        "stm": {
            "sessions": stm_sessions,
            "chunks": stm_chunks,
            "turns": stm_turns,
            "entities": stm_entities,
        },
        "ltm": {
            "sessions": ltm_sessions,
            "entities": ltm_entities,
            "topics": ltm_topics,
            "chunks": ltm_chunks,
            "turns": ltm_turns,
        },
    }


async def namespaces(_: Request) -> JSONResponse:
    """List available namespaces with per-namespace STM + LTM metadata."""
    try:
        client = build_arcadedb_client(get_settings())
        await client.connect()
        try:
            dbs = await client.list_databases()
            names = namespaces_from_databases(dbs)
            metadata = await asyncio.gather(*(_namespace_stats(client, ns) for ns in names))
        finally:
            await client.close()
        return JSONResponse(
            {"namespaces": metadata, "default_namespace": get_settings().namespace}
        )
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def healthz(_: Request) -> JSONResponse:
    """Liveness probe — process is up (no backend calls)."""
    return JSONResponse({"status": "ok"})


async def readyz(_: Request) -> JSONResponse:
    """Readiness probe — backend reachable (503 if not)."""
    try:
        report = await get_runtime().doctor()
        ok = report.get("arcadedb_stm") is True and report.get("arcadedb_ltm") is True
        return JSONResponse(report, status_code=200 if ok else 503)
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def health(_: Request) -> JSONResponse:
    """Return backend connectivity status."""
    try:
        return JSONResponse(await get_runtime().doctor())
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def stats(request: Request) -> JSONResponse:
    """Return STM and LTM memory statistics for the requested namespace."""
    try:
        session_id = request.query_params.get("session_id", "")
        namespace = request.query_params.get("namespace", "")
        client = build_arcadedb_client(get_settings())
        await client.connect()
        try:
            return JSONResponse(await _namespace_stats(client, namespace, session_id))
        finally:
            await client.close()
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def sessions(request: Request) -> JSONResponse:
    """Return known session ids from STM and LTM stores for the requested namespace."""
    namespace = request.query_params.get("namespace", "")
    values: set[str] = set()
    errors: list[str] = []

    try:
        stm = build_arcadedb_stm_store(get_settings(), namespace)
        await stm.connect()
        try:
            rows = await stm.client.query(
                stm.database,
                "MATCH (n:ENTITY) WHERE n.session_id IS NOT NULL "
                "RETURN DISTINCT n.session_id AS id ORDER BY id",
            )
            values.update(_string(row["id"]) for row in rows if row.get("id"))
        finally:
            await stm.close()
    except Exception as exc:
        errors.append(f"stm: {exc}")

    try:
        ltm = build_arcadedb_ltm_store(get_settings(), namespace)
        await ltm.connect()
        try:
            rows = await ltm._qry("MATCH (s:SESSION) RETURN DISTINCT s.id AS id ORDER BY id")
            values.update(_string(row["id"]) for row in rows if row.get("id"))
        finally:
            await ltm.close()
    except Exception as exc:
        errors.append(f"ltm: {exc}")

    return JSONResponse({"sessions": sorted(v for v in values if v), "errors": errors})


async def graph(request: Request) -> JSONResponse:
    """Return normalized nodes and links for the requested graph scope."""
    scope = request.query_params.get("scope", "ltm").lower()
    session_id = request.query_params.get("session_id", "")
    namespace = request.query_params.get("namespace", "")
    limit = max(10, min(int(request.query_params.get("limit", "200")), 1000))

    try:
        if scope == "stm":
            if not session_id:
                return JSONResponse(
                    {
                        "error": "session_id_required",
                        "message": "STM graph views require a session_id.",
                    },
                    status_code=400,
                )
            stm = build_arcadedb_stm_store(get_settings(), namespace)
            await stm.connect()
            try:
                raw = await stm.get_session_graph(session_id)
            finally:
                await stm.close()
            return JSONResponse(_normalise_stm_graph(raw, limit))

        if scope != "ltm":
            return JSONResponse(
                {"error": "invalid_scope", "message": "scope must be 'stm' or 'ltm'."},
                status_code=400,
            )
        return JSONResponse(await _ltm_graph(session_id, limit, namespace))
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def recall(request: Request) -> JSONResponse:
    """Return ranked recall hits for a query."""
    query = request.query_params.get("q", "").strip()
    session_id = request.query_params.get("session_id", "")
    scope = request.query_params.get("scope", "both")
    namespace = request.query_params.get("namespace", "")
    top_k = max(1, min(int(request.query_params.get("top_k", "5")), 25))
    if not query:
        return JSONResponse({"hits": []})
    if scope in {"stm", "both"} and not session_id:
        return JSONResponse(
            {"error": "session_id_required", "message": "STM recall requires a session_id."},
            status_code=400,
        )
    try:
        result = await get_runtime(namespace).recall_memory(
            query, session_id=session_id, top_k=top_k, scope=scope
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def ingest(request: Request) -> JSONResponse:
    """Ingest a markdown/text file into STM via multipart form upload."""
    try:
        form = await request.form()
    except Exception as exc:
        return JSONResponse({"error": "bad_request", "message": str(exc)}, status_code=400)

    session_id = str(form.get("session_id", "")).strip()
    if not session_id:
        return JSONResponse(
            {"error": "session_id_required", "message": "session_id is required to ingest."},
            status_code=400,
        )

    file_field = form.get("file")
    if file_field is None:
        return JSONResponse(
            {"error": "file_required", "message": "A file field is required."},
            status_code=400,
        )

    namespace = str(form.get("namespace", "")).strip()
    title = str(form.get("title", "")).strip()
    try:
        raw_bytes = await file_field.read()  # type: ignore[union-attr]
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return JSONResponse(
            {"error": "read_error", "message": f"Could not read file: {exc}"}, status_code=400
        )

    try:
        result = await get_runtime(namespace).ingest_memory(
            text, session_id=session_id, title=title
        )
        return JSONResponse(
            {
                "file_id": result["file_id"],
                "chunk_count": result["chunks"],
                "entity_count": result["entities"],
                "elapsed_s": result["elapsed_s"],
                "errors": result["errors"],
            }
        )
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def flush_session(request: Request) -> JSONResponse:
    """Promote a session's STM to LTM and return flush statistics."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(body.get("session_id", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    if not session_id:
        return JSONResponse(
            {"error": "session_id_required", "message": "session_id is required to flush."},
            status_code=400,
        )

    try:
        result = await get_runtime(namespace).flush_memory(session_id)
        return JSONResponse(
            {
                "session_id": session_id,
                "nodes_written": result.get("nodes", 0),
                "edges_written": result.get("edges", 0),
                "chunks_promoted": result.get("chunks", 0),
                "turns_promoted": result.get("turns", 0),
                "merged_entities": result.get("merged_entities", 0),
                "community_count": 0,
                "modularity": 0.0,
                "ran_iterations": 0,
                "topics_created": result.get("topics_created", 0),
                "elapsed_s": result.get("elapsed_s", 0.0),
                "errors": result.get("errors", []),
            }
        )
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


def _routes() -> list[Any]:
    frontend_dir = _frontend_dir()
    routes: list[Any] = [
        Route("/healthz", healthz),
        Route("/readyz", readyz),
        Route("/api/health", health),
        Route("/api/namespaces", namespaces),
        Route("/api/stats", stats),
        Route("/api/sessions", sessions),
        Route("/api/graph", graph),
        Route("/api/recall", recall),
        Route("/api/ingest", ingest, methods=["POST"]),
        Route("/api/flush", flush_session, methods=["POST"]),
    ]
    if frontend_dir.exists():
        routes.append(
            Mount("/", app=StaticFiles(directory=frontend_dir, html=True), name="frontend")
        )
    else:
        routes.append(
            Route("/", lambda _: Response("frontend directory not found", status_code=404))
        )
    return routes


app = Starlette(debug=False, routes=_routes())


def main() -> None:
    """Run the local Memory Mirror web server."""
    get_settings().check_production_secrets()
    host = os.environ.get("GRAPHKNOWS_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPHKNOWS_WEB_PORT", DEFAULT_PORT))
    uvicorn.run("graphknows.server.web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
