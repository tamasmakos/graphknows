"""Helpers for writing reusable pipeline diagnostic artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def diagnostics_enabled(config: Dict[str, Any] | None) -> bool:
    if not config:
        return False
    extraction_cfg = config.get("extraction", {})
    if hasattr(extraction_cfg, "model_dump"):
        extraction_cfg = extraction_cfg.model_dump()
    return bool(extraction_cfg.get("diagnostic_mode", False))


def diagnostic_dir(config: Dict[str, Any]) -> Path:
    extraction_cfg = config.get("extraction", {})
    infra_cfg = config.get("infra", {})
    if hasattr(extraction_cfg, "model_dump"):
        extraction_cfg = extraction_cfg.model_dump()
    if hasattr(infra_cfg, "model_dump"):
        infra_cfg = infra_cfg.model_dump()

    output_dir = Path(infra_cfg.get("output_dir", "output"))
    subdir = extraction_cfg.get("diagnostic_output_subdir", "diagnostics")
    return output_dir / subdir


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "diagnostic"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_diagnostic_json(config: Dict[str, Any], name: str, payload: Any) -> str | None:
    if not diagnostics_enabled(config):
        return None

    out_dir = diagnostic_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_sanitize_name(name)}.json"
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
