"""URL helpers for the local Memory Mirror UI."""

from __future__ import annotations

import os
from urllib.parse import urlencode

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8800


def mirror_url(session_id: str = "", scope: str = "ltm") -> str:
    """Return the public Memory Mirror URL for the running web server."""
    host = os.environ.get("GRAPHKNOWS_WEB_PUBLIC_HOST") or os.environ.get(
        "GRAPHKNOWS_WEB_HOST", DEFAULT_HOST
    )
    # GRAPHKNOWS_WEB_PUBLIC_PORT lets operators set the externally visible port
    # (e.g. 80 via a reverse proxy) independently from the internal bind port.
    port_str = (
        os.environ.get("GRAPHKNOWS_WEB_PUBLIC_PORT")
        or os.environ.get("GRAPHKNOWS_WEB_PORT")
    )
    port = int(port_str) if port_str else DEFAULT_PORT
    params = {"scope": scope or "ltm"}
    if session_id:
        params["session_id"] = session_id
    # Omit the port from the URL when it is the default for the scheme (80).
    authority = host if port == 80 else f"{host}:{port}"
    return f"http://{authority}/?{urlencode(params)}"
