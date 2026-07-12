"""Persistent judge cache.

The LLM judge is non-deterministic even at temperature 0 (~25% of borderline
triples flip between runs). Caching each (question, gold, answer, prompt
version) verdict on disk makes an unchanged answer contribute exactly 0 to a
measured A/B delta, which is what makes small tuning deltas readable.

Disable with ``GRAPHKNOWS_JUDGE_CACHE=0``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_CACHE_DIR = Path("evaluation/results/.judge_cache")


def _enabled() -> bool:
    return os.environ.get("GRAPHKNOWS_JUDGE_CACHE", "1") != "0"


def _key(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
    return digest


class JudgeCache:
    """One cache namespace per benchmark judge (keyed into the shared dir)."""

    def __init__(self, namespace: str, cache_dir: Path = _CACHE_DIR) -> None:
        self._dir = cache_dir / namespace
        if _enabled():
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, *parts: str) -> dict[str, Any] | None:
        if not _enabled():
            return None
        path = self._dir / f"{_key(parts)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, value: dict[str, Any], *parts: str) -> None:
        if not _enabled():
            return
        path = self._dir / f"{_key(parts)}.json"
        # Cache is best-effort; never fail a run over it. Write to a unique temp
        # file then atomically rename so a concurrent reader never sees a
        # half-written file when many judges run in parallel.
        with contextlib.suppress(OSError):
            tmp = path.with_suffix(f".{os.getpid()}.{id(value):x}.tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
