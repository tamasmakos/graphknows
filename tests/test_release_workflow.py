"""Release workflow contract: what makes a tag push actually publish.

These invariants exist because nothing in the repo publishes the package
today — a release is still a hand-run `uv build` + upload from a laptop.
`.github/workflows/ci.yml` was already made `workflow_call`-able (see
test_ci_workflow.py) specifically so a tag-triggered release workflow could
reuse it, but no caller exists yet:

  * The release workflow must fire on `vX.Y.Z` tag pushes only — never on
    pull_request or branch pushes, which would publish on every commit.
  * It must call `ci.yml` by reference (`uses:`) rather than re-implementing
    `uv sync` / `pytest`, otherwise the two checklists drift out of sync.
  * Publishing must depend on that call succeeding (`needs:`), and must not
    build a second time — it consumes the `dist` artifact CI already built
    and uploaded, so what ships is exactly what CI tested.
  * The tag/version check must read the literal out of
    `graphknows/_version.py` as text, never `import graphknows` — importing
    the package to release it is exactly backwards (it may not even be
    installable yet) and the version must never be hand-restated in the
    workflow, or the two will eventually disagree.
  * Publishing must go through PyPI's Trusted Publisher (OIDC), which needs
    `id-token: write` scoped to the publish job only (never at the top-level
    `permissions:` block, which would over-grant it to every job) and a
    SHA-pinned `pypa/gh-action-pypi-publish` — a mutable tag on a publishing
    action is a supply-chain hole, and a stored `PYPI_API_TOKEN` is exactly
    the long-lived secret Trusted Publisher exists to remove.
  * The release run needs its own concurrency group with cancellation off —
    reusing CI's group, or leaving cancel-in-progress on, risks a superseded
    run cancelling the actual publish.

A second, real defect rides along: docs/versioning.md claims the version is
single-sourced from `graphknows/__init__.py`, while
`[tool.hatch.version] path = "graphknows/_version.py"` is what the build
backend actually reads (`__init__.py` merely re-exports it). Anyone
automating a release from that sentence automates the wrong file.

Read as text with pathlib + re, not PyYAML: this project does not declare a
yaml dependency, and YAML 1.1 parses the bare `on:` key as the boolean
`True`, which makes structural assertions on it worse than a regex over text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / ".github" / "workflows" / "release.yml"
VERSION_MODULE = REPO_ROOT / "graphknows" / "_version.py"
VERSIONING_DOC = REPO_ROOT / "docs" / "versioning.md"


def _release_text() -> str:
    assert RELEASE.is_file(), (
        f"{RELEASE}: no tag-triggered release workflow exists — pushing a tag publishes nothing"
    )
    return RELEASE.read_text(encoding="utf-8")


def test_triggers_only_on_version_tag_pushes() -> None:
    """The workflow must fire on `vX.Y.Z` tags only — no pull_request, no
    branch push — otherwise it publishes on every ordinary commit."""
    text = _release_text()
    assert re.search(r"tags:\s*\n?\s*-?\s*\[?\s*[\"']?v\*?\.?\[?0-9", text), (
        f"{RELEASE}: no `tags:` trigger matching a v-prefixed three-part version pattern found"
    )
    assert not re.search(r"^\s*pull_request:", text, re.MULTILINE), (
        f"{RELEASE}: has a `pull_request:` trigger — a release workflow "
        "must only fire on tag pushes"
    )
    assert not re.search(r"^\s*branches:", text, re.MULTILINE), (
        f"{RELEASE}: has a `branches:` trigger — a release workflow must only fire on tag pushes"
    )


def test_calls_ci_by_reference_not_by_copy() -> None:
    """Must `uses: ./.github/workflows/ci.yml` with `secrets: inherit`, and
    must never re-declare CI's own steps."""
    text = _release_text()
    assert re.search(r"uses:\s*\./\.github/workflows/ci\.yml", text), (
        f"{RELEASE}: does not call ci.yml via `uses: ./.github/workflows/ci.yml`"
    )
    assert "secrets: inherit" in text, f"{RELEASE}: missing `secrets: inherit`"
    assert "uv sync" not in text, (
        f"{RELEASE}: re-declares `uv sync` instead of reusing ci.yml via `uses:`"
    )
    assert "pytest" not in text, (
        f"{RELEASE}: re-declares `pytest` instead of reusing ci.yml via `uses:`"
    )


def test_publish_depends_on_the_ci_call() -> None:
    """The publish job must `needs:` the job that calls ci.yml, otherwise a
    broken CI run does not block publishing."""
    text = _release_text()
    assert re.search(r"^\s*needs:", text, re.MULTILINE), (
        f"{RELEASE}: no `needs:` line — publishing does not depend on the CI call"
    )


def test_publish_builds_nothing_and_consumes_cis_artifact() -> None:
    """Publishing must not build a second time — it must download the exact
    `dist` artifact CI already built and uploaded."""
    text = _release_text()
    assert "uv build" not in text, (
        f"{RELEASE}: runs `uv build` — publishing must reuse CI's build, not repeat it"
    )
    assert "python -m build" not in text, (
        f"{RELEASE}: runs `python -m build` — publishing must reuse CI's build, not repeat it"
    )
    assert "actions/download-artifact" in text, (
        f"{RELEASE}: never downloads CI's build output (no `actions/download-artifact`)"
    )
    assert re.search(r"name:\s*dist", text), (
        f"{RELEASE}: `actions/download-artifact` step does not target `name: dist`"
    )


def test_version_check_reads_source_text_not_the_import() -> None:
    """The tag/version assertion must read the literal out of
    `graphknows/_version.py` as text — importing the package to release it
    is exactly backwards."""
    text = _release_text()
    assert "_version.py" in text, (
        f"{RELEASE}: does not reference `graphknows/_version.py` / `_version.py`"
    )
    assert "import graphknows" not in text, (
        f"{RELEASE}: contains `import graphknows` — the version check must "
        "read source text, not import the package"
    )
    assert "from graphknows" not in text, (
        f"{RELEASE}: contains `from graphknows` — the version check must "
        "read source text, not import the package"
    )


def test_version_is_never_hand_restated_in_the_workflow() -> None:
    """The literal version string from `graphknows/_version.py` must not
    appear in the workflow — restating it invites drift."""
    version_text = VERSION_MODULE.read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', version_text, re.MULTILINE)
    assert match is not None, f'{VERSION_MODULE}: no `__version__ = "..."` literal found'
    version = match.group(1)
    text = _release_text()
    assert version not in text, (
        f"{RELEASE}: hand-restates the current version {version!r} — "
        f"it must be read from {VERSION_MODULE}, not duplicated"
    )


def test_publishes_via_trusted_publisher_sha_pinned_no_stored_token() -> None:
    """Publishing must go through PyPI's Trusted Publisher (OIDC) with the
    publish action SHA-pinned (a mutable tag is a supply-chain hole) and no
    stored API token."""
    text = _release_text()
    assert re.search(r"uses:\s*pypa/gh-action-pypi-publish@([0-9a-f]{40})\b", text), (
        f"{RELEASE}: `pypa/gh-action-pypi-publish` is not pinned to a 40-char "
        "commit SHA — a mutable tag (e.g. @v1.2.3) is a supply-chain risk"
    )
    assert "secrets.PYPI" not in text, (
        f"{RELEASE}: references `secrets.PYPI*` — publishing must use "
        "Trusted Publisher (OIDC), not a stored token"
    )
    assert "password:" not in text, (
        f"{RELEASE}: has a `password:` line — publishing must use Trusted "
        "Publisher (OIDC), not a stored token"
    )
    assert "api_token" not in text, (
        f"{RELEASE}: references `api_token` — publishing must use Trusted "
        "Publisher (OIDC), not a stored token"
    )


def test_id_token_write_is_scoped_to_the_publish_job_only() -> None:
    """`id-token: write` must appear exactly once, indented as a job-level
    `permissions:` entry — never granted at the workflow's top-level
    `permissions:` block, which would over-grant it to every job."""
    text = _release_text()
    occurrences = [line for line in text.splitlines() if re.search(r"id-token:\s*write", line)]
    assert len(occurrences) == 1, (
        f"{RELEASE}: expected exactly one `id-token: write` line, found {len(occurrences)}"
    )
    line = occurrences[0]
    leading_spaces = len(line) - len(line.lstrip(" "))
    assert leading_spaces >= 6, (
        f"{RELEASE}: `id-token: write` is indented {leading_spaces} spaces — "
        f"expected a job-level `permissions:` block (>=6 spaces), got: {line.strip()!r}"
    )
    top_level_permissions = re.search(r"^permissions:\n((?:^\s{2}.*\n?)*)", text, re.MULTILINE)
    assert top_level_permissions is not None, f"{RELEASE}: no top-level `permissions:` block found"
    top_block = top_level_permissions.group(1)
    assert "contents: read" in top_block, (
        f"{RELEASE}: top-level `permissions:` block does not grant `contents: read`"
    )
    assert "id-token: write" not in top_block, (
        f"{RELEASE}: top-level `permissions:` block grants `id-token: write` — "
        "it must be scoped to the publish job only"
    )


def test_has_its_own_non_cancelling_concurrency_group() -> None:
    """The release run must use its own concurrency group (not CI's) with
    cancellation off, so a superseded run never cancels an actual publish."""
    text = _release_text()
    assert re.search(r"^\s*concurrency:", text, re.MULTILINE), (
        f"{RELEASE}: no `concurrency:` block found"
    )
    group_match = re.search(r"^\s*group:\s*(.+)$", text, re.MULTILINE)
    assert group_match is not None, f"{RELEASE}: `concurrency:` block has no `group:` line"
    assert group_match.group(1).strip() != "ci-${{ github.ref }}", (
        f"{RELEASE}: reuses CI's concurrency group `ci-${{{{ github.ref }}}}` — "
        "the release run must have its own group"
    )
    assert re.search(r"^\s*cancel-in-progress:\s*false\s*$", text, re.MULTILINE), (
        f"{RELEASE}: `cancel-in-progress: false` not set — a superseded run "
        "could cancel an in-flight publish"
    )


def test_docs_name_the_module_the_build_backend_actually_reads() -> None:
    """docs/versioning.md must name `graphknows/_version.py` — the file
    `[tool.hatch.version]` actually reads — and stop claiming the version is
    single-sourced from `graphknows/__init__.py`, which only re-exports it."""
    text = VERSIONING_DOC.read_text(encoding="utf-8")
    assert "graphknows/_version.py" in text, (
        f"{VERSIONING_DOC}: does not name `graphknows/_version.py`, the file "
        "the build backend actually reads"
    )
    assert "single-sourced from `graphknows/__init__.py`" not in text, (
        f"{VERSIONING_DOC}: still claims the version is single-sourced from "
        "`graphknows/__init__.py` — hatch reads `graphknows/_version.py`, and "
        "automating a release from this sentence bumps the wrong file"
    )


def test_docs_describe_the_tag_push_release_mechanism() -> None:
    """The release checklist must document pushing the tag, the one-time
    Trusted Publisher registration, and the fix-forward rule (a tag is never
    deleted/re-pushed) — otherwise the new mechanism is undocumented."""
    text = VERSIONING_DOC.read_text(encoding="utf-8")
    assert re.search(r"git push.*tag|push the tag", text, re.IGNORECASE), (
        f"{VERSIONING_DOC}: does not mention pushing the release tag"
    )
    assert "Trusted Publisher" in text, (
        f"{VERSIONING_DOC}: does not mention the one-time Trusted Publisher registration"
    )
    assert re.search(r"fix[- ]forward", text, re.IGNORECASE), (
        f"{VERSIONING_DOC}: does not document the fix-forward rule"
    )
    assert re.search(r"never.{0,40}re-push|re-push.{0,40}never", text, re.IGNORECASE | re.DOTALL), (
        f"{VERSIONING_DOC}: does not state that a tag is never re-pushed/deleted"
    )
