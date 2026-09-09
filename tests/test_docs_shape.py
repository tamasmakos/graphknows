"""Docs that name code must match the code they name."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_name_no_absent_symbols() -> None:
    """Docs must not describe code that does not exist (FR-026, SC-017).

    Greps `graphknows/` for each symbol FR-026 named as removed — the
    `community_boost` / `shape_score` ranking stage and `ArcadeDBSTMStore` /
    `ArcadeDBLTMStore` — so the removal claim is checked against the code, not
    assumed. Then greps `docs/architecture.md` for the same symbols, plus
    "the web server", a prose claim rather than a single token.
    """
    absent_symbols = {"community_boost", "shape_score", "ArcadeDBSTMStore", "ArcadeDBLTMStore"}

    code_text = "\n".join(
        p.read_text(encoding="utf-8") for p in (REPO_ROOT / "graphknows").rglob("*.py")
    )
    for symbol in absent_symbols:
        assert symbol not in code_text, (
            f"{symbol} exists in graphknows/ — the FR-026 doc claim is stale"
        )

    for doc_name in ("architecture.md",):
        doc_path = REPO_ROOT / "docs" / doc_name
        text = doc_path.read_text(encoding="utf-8")
        for symbol in absent_symbols:
            assert symbol not in text, f"{doc_name} still names {symbol}, absent from graphknows/"
        assert "web server" not in text.lower(), f"{doc_name} still names the web server"


def test_torch_backend_guidance_is_consistent() -> None:
    """`TORCH_BACKEND` guidance is one value, stated once, referenced everywhere (FR-044).

    Also: the changelog must not describe PyPI Trusted Publisher OIDC publishing,
    which `release.yml` does not do — it attaches the wheel to a GitHub Release.
    """
    dockerfile_text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    compose_text = (REPO_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

    pattern = re.compile(r"TORCH_BACKEND[=:]\s*(cu\d+)")
    recommended = (
        set(pattern.findall(dockerfile_text))
        | set(pattern.findall(readme_text))
        | set(pattern.findall(compose_text))
    )
    assert recommended == {"cu126"}, (
        f"TORCH_BACKEND guidance disagrees across Dockerfile/README.md/docker-compose.yaml: "
        f"{sorted(recommended)}"
    )

    changelog_text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_workflow_text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "pypi" not in release_workflow_text.lower(), (
        "release.yml now publishes to PyPI — CHANGELOG.md's correction is stale"
    )
    assert "to pypi" not in changelog_text.lower(), (
        "CHANGELOG.md still claims release.yml publishes to PyPI"
    )
    assert "trusted publisher oidc flow" not in changelog_text.lower(), (
        "CHANGELOG.md still claims release.yml uses the Trusted Publisher OIDC flow"
    )
