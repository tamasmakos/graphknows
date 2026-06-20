"""Shared Hugging Face dataset helpers for evaluation benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

HF_ROWS = "https://datasets-server.huggingface.co/rows"


def read_json_url(url: str) -> Any:
    """Read JSON from a URL."""
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_if_missing(url: str, path: Path) -> None:
    """Download a URL into a local ignored data path if needed."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())


def fetch_dataset_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch rows from the Hugging Face datasets-server."""
    params = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    payload = read_json_url(f"{HF_ROWS}?{params}")
    rows = [item["row"] for item in payload.get("rows", [])]
    return rows, int(payload["num_rows_total"])
