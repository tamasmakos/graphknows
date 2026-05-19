"""Provenance and reproducibility utilities."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, Iterable, Optional


_SECRET_KEYWORDS = ("password", "api_key", "token", "secret", "key")


def _is_secret_key(key: str) -> bool:
    return any(token in key.lower() for token in _SECRET_KEYWORDS)


def _redact(value: Any) -> Any:
    if hasattr(value, "get_secret_value"):
        return "REDACTED"
    return value


def redact_secrets(data: Any) -> Any:
    """Recursively redact secrets from dicts/lists."""
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and _is_secret_key(k):
                cleaned[k] = "REDACTED"
                continue
            cleaned[k] = redact_secrets(_redact(v))
        return cleaned
    if isinstance(data, list):
        return [redact_secrets(_redact(v)) for v in data]
    return _redact(data)


def _safe_serialize(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe_serialize(v) for v in obj]
        return str(obj)


def get_git_revision(repo_root: str) -> Optional[str]:
    """Read git HEAD without invoking subprocess."""
    head_path = os.path.join(repo_root, ".git", "HEAD")
    if not os.path.exists(head_path):
        return None
    try:
        with open(head_path, "r", encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref_path = head.split(" ", 1)[1].strip()
            full_ref_path = os.path.join(repo_root, ".git", ref_path)
            if os.path.exists(full_ref_path):
                with open(full_ref_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        return head
    except Exception:
        return None


def _get_package_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def collect_environment_info() -> Dict[str, Any]:
    """Collect basic runtime environment information."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "networkx",
        "matplotlib",
        "seaborn",
        "torch",
        "pykeen",
        "sentence-transformers",
        "pydantic",
    ]
    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "package_versions": _get_package_versions(packages),
    }


def write_pipeline_config_snapshot(output_dir: str, settings: Any) -> str:
    """Write a sanitized config snapshot (JSON-compatible YAML)."""
    if hasattr(settings, "model_dump"):
        config = settings.model_dump()
    else:
        config = settings.dict()  # type: ignore[no-any-return]
    config = redact_secrets(config)
    config_path = os.path.join(output_dir, "pipeline_config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def write_analysis_run_manifest(
    output_dir: str,
    settings: Any,
    stage: str,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a run manifest with provenance and summary info."""
    manifest: Dict[str, Any] = {
        "stage": stage,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "timestamp": datetime.now().isoformat(),
        "environment": collect_environment_info(),
        "git_revision": get_git_revision(os.getcwd()),
    }
    if extra:
        manifest["extra"] = _safe_serialize(extra)
    if hasattr(settings, "model_dump"):
        settings_dict = settings.model_dump()
    else:
        settings_dict = settings.dict()  # type: ignore[no-any-return]
    manifest["settings_summary"] = _safe_serialize(redact_secrets(settings_dict))
    manifest_path = os.path.join(output_dir, "analysis_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_safe_serialize(manifest), f, indent=2, ensure_ascii=False)
    return manifest_path
