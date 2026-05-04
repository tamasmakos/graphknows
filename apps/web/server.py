import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8002")
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title="Graphknows Web")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global _client
    _client = httpx.AsyncClient(base_url=AGENT_URL, timeout=60.0)


@app.on_event("shutdown")
async def shutdown():
    if _client:
        await _client.aclose()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    url = f"/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    try:
        resp = await _client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except httpx.ConnectError:
        return Response(
            content='{"error":"agent_offline"}',
            status_code=503,
            media_type="application/json",
        )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
