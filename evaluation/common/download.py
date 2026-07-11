"""Dataset download helper (streamed, resumable-by-rerun)."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def download_file(url: str, dest: Path, *, timeout: float = 600.0) -> Path:
    """Stream *url* to *dest* (skips when the file already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s → %s", url, dest)
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest
