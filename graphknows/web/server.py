"""Tiny local web server for exploring graphknows memory graphs."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from graphknows.config.health import check_backends, get_memory_stats
from graphknows.mcp._state import get_mg_store, get_neo_store, get_pg_store, get_settings
from graphknows.web.links import DEFAULT_PORT


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
        OPTIONAL MATCH (e)-[:IN_COMMUNITY|IN_TOPIC]->(t:TOPIC)
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
        MATCH (a)-[r:CO_OCCURS]-(b)
        WITH DISTINCT a, b, r LIMIT $limit
        RETURN coalesce(a.id, a.name) AS source,
               coalesce(b.id, b.name) AS target,
               coalesce(r.label, type(r)) AS label,
               coalesce(r.weight, 1) AS weight
        """
        return node_cypher, edge_cypher, {"session_id": session_id, "limit": limit}

    node_cypher = """
    MATCH (e:ENTITY)
    OPTIONAL MATCH (e)-[:IN_COMMUNITY|IN_TOPIC]->(t:TOPIC)
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
    MATCH (a:ENTITY)-[r]->(b:ENTITY)
    WHERE a <> b
    WITH a, b, type(r) AS label, coalesce(r.weight, 1) AS weight LIMIT $limit
    RETURN coalesce(a.id, a.name) AS source,
           coalesce(b.id, b.name) AS target,
           label,
           weight
    """
    return node_cypher, edge_cypher, {"limit": limit}


async def _ltm_graph(session_id: str, limit: int) -> dict[str, Any]:
    neo = get_neo_store()
    try:
        node_cypher, edge_cypher, params = _ltm_queries(session_id, limit)
        node_rows, edge_rows = await asyncio.gather(
            neo._arun(node_cypher, **params), neo._arun(edge_cypher, **params)
        )
    finally:
        neo.close()

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


async def health(_: Request) -> JSONResponse:
    """Return backend connectivity status."""
    try:
        return JSONResponse(await check_backends(get_settings()))
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def stats(request: Request) -> JSONResponse:
    """Return STM and LTM memory statistics."""
    try:
        session_id = request.query_params.get("session_id", "")
        return JSONResponse(await get_memory_stats(get_settings(), session_id=session_id))
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


async def sessions(_: Request) -> JSONResponse:
    """Return known session ids from STM and LTM stores."""
    values: set[str] = set()
    errors: list[str] = []

    try:
        mg = get_mg_store()

        def _read_stm_sessions() -> list[Any]:
            with mg._driver.session() as session:
                return list(
                    session.run(
                        "MATCH (n:ENTITY) "
                        "WHERE n.session_id IS NOT NULL "
                        "RETURN DISTINCT n.session_id AS id ORDER BY id"
                    )
                )

        rows = await asyncio.to_thread(_read_stm_sessions)
        values.update(_string(row["id"]) for row in rows if row.get("id"))
        mg.close()
    except Exception as exc:
        errors.append(f"stm: {exc}")

    try:
        neo = get_neo_store()
        rows = await neo._arun("MATCH (s:SESSION) RETURN DISTINCT s.id AS id ORDER BY id")
        values.update(_string(row["id"]) for row in rows if row.get("id"))
        neo.close()
    except Exception as exc:
        errors.append(f"ltm: {exc}")

    return JSONResponse({"sessions": sorted(v for v in values if v), "errors": errors})


async def graph(request: Request) -> JSONResponse:
    """Return normalized nodes and links for the requested graph scope."""
    scope = request.query_params.get("scope", "ltm").lower()
    session_id = request.query_params.get("session_id", "")
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
            mg = get_mg_store()
            try:
                raw = await mg.get_session_graph(session_id)
            finally:
                mg.close()
            return JSONResponse(_normalise_stm_graph(raw, limit))

        if scope != "ltm":
            return JSONResponse(
                {"error": "invalid_scope", "message": "scope must be 'stm' or 'ltm'."},
                status_code=400,
            )
        return JSONResponse(await _ltm_graph(session_id, limit))
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


def _normalise_hits(hits: list[Any], weight: float) -> None:
    """Min-max normalise hit scores in place, then apply weight factor."""
    if not hits:
        return
    scores = [h.score for h in hits]
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    for h in hits:
        h.score = weight * ((h.score - mn) / rng if rng else 1.0)


def _run_recall(
    scope: str,
    query: str,
    session_id: str,
    top_k: int,
) -> dict[str, Any]:
    """Synchronous recall across STM/LTM stores. Returns serialisable hits dict."""
    from graphknows.memory.stm._async_bridge import _run as _bg_run
    from graphknows.memory.stores.embedder import embed_one

    def _clean(text: str) -> str:
        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _fmt(results: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "score": round(r.score, 6),
                "cosine_score": round(getattr(r, "cosine_score", r.score), 6),
                "type": r.source_type,
                "text": _clean(r.text),
                "session_id": r.provenance.get("session_id", session_id),
                "entities": list(r.entity_names or []),
            }
            for r in results
        ]

    stm_hits: list[Any] = []
    ltm_hits: list[Any] = []

    if scope in ("stm", "both"):
        from graphknows.memory.stm.recall import recall_short_term

        mg = get_mg_store()
        pg = get_pg_store()
        _bg_run(pg.connect())
        stm_hits = recall_short_term(
            mg, pg, query, session_id=session_id, top_k=top_k, embed_fn=embed_one
        )
        mg.close()

    if scope in ("ltm", "both"):
        from graphknows.memory.query.recall_ltm import recall_long_term

        pg2 = get_pg_store()
        neo = get_neo_store()
        _bg_run(pg2.connect())
        ltm_hits = recall_long_term(
            neo, pg2, query, top_k=top_k, session_filter=session_id or None, embed_fn=embed_one
        )
        neo.close()

    if scope == "both":
        _normalise_hits(stm_hits, 0.6)
        _normalise_hits(ltm_hits, 0.4)
        combined = sorted(stm_hits + ltm_hits, key=lambda h: h.score, reverse=True)
        return {"hits": _fmt(combined[:top_k])}

    return {"hits": _fmt(stm_hits if scope == "stm" else ltm_hits)}


async def recall(request: Request) -> JSONResponse:
    """Return ranked recall hits for a query."""
    query = request.query_params.get("q", "").strip()
    session_id = request.query_params.get("session_id", "")
    scope = request.query_params.get("scope", "both")
    top_k = max(1, min(int(request.query_params.get("top_k", "5")), 25))
    if not query:
        return JSONResponse({"hits": []})
    if scope in {"stm", "both"} and not session_id:
        return JSONResponse(
            {"error": "session_id_required", "message": "STM recall requires a session_id."},
            status_code=400,
        )
    try:
        result = await asyncio.to_thread(_run_recall, scope, query, session_id, top_k)
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

    title = str(form.get("title", "")).strip()
    try:
        raw_bytes = await file_field.read()  # type: ignore[union-attr]
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return JSONResponse(
            {"error": "read_error", "message": f"Could not read file: {exc}"}, status_code=400
        )

    try:
        from graphknows.config.builders.stores import build_stm_service

        service = await asyncio.to_thread(build_stm_service, get_settings())
        result = await asyncio.to_thread(service.ingest, text, session_id, title)
        return JSONResponse(
            {
                "file_id": result.file_id,
                "chunk_count": result.chunk_count,
                "entity_count": result.entity_count,
                "elapsed_s": round(result.elapsed_s, 3),
                "errors": list(result.errors),
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
    if not session_id:
        return JSONResponse(
            {"error": "session_id_required", "message": "session_id is required to flush."},
            status_code=400,
        )

    opts_raw = body.get("options", {}) or {}

    try:
        from graphknows.config.builders.stores import build_promotion_pipeline
        from graphknows.memory.models.promotion import PromotionRequest

        request = PromotionRequest(
            session_id=session_id,
            clear_stm=bool(opts_raw.get("clear_stm", True)),
            run_gds=bool(opts_raw.get("run_gds", True)),
            merge_entities=bool(opts_raw.get("merge_entities", True)),
        )
        pipeline = await asyncio.to_thread(build_promotion_pipeline, get_settings())
        result = await pipeline.promote(request)
        gds = result.gds_result
        return JSONResponse(
            {
                "session_id": result.session_id,
                "nodes_written": result.nodes_written,
                "edges_written": result.edges_written,
                "chunks_promoted": result.chunks_promoted,
                "turns_promoted": result.turns_promoted,
                "merged_entities": result.merged_entities,
                "community_count": gds.community_count if gds else 0,
                "modularity": round(gds.modularity, 4) if gds else 0.0,
                "ran_iterations": gds.ran_iterations if gds else 0,
                "topics_created": result.topics_created,
                "elapsed_s": round(result.elapsed_s, 2),
                "errors": result.errors,
            }
        )
    except Exception as exc:
        return JSONResponse(_error_payload(exc), status_code=503)


def _routes() -> list[Any]:
    frontend_dir = _frontend_dir()
    routes: list[Any] = [
        Route("/api/health", health),
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
    host = os.environ.get("GRAPHKNOWS_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPHKNOWS_WEB_PORT", DEFAULT_PORT))
    uvicorn.run("graphknows.web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
