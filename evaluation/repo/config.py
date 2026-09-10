"""Repository-transcript row configuration.

The corpus is this repository's own agent transcripts, so there is no download
step and no per-question haystack: one transcript is one document. Session
content is not committed: a gold record names a transcript, and it is read from
``data_root`` when it is the committed fixture and from ``sessions_root`` — the
harness's own directory for this checkout — otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from evaluation.common.config import BaseEvalConfig


def claude_code_sessions_root() -> Path:
    """Where the Claude Code harness keeps this checkout's session transcripts.

    The harness names a project directory after its working directory, with
    every non-alphanumeric character replaced by a dash.
    """
    project = re.sub(r"[^A-Za-z0-9]", "-", str(Path.cwd()))
    return Path.home() / ".claude" / "projects" / project


class RepoConfig(BaseEvalConfig):
    """The repository-transcript evaluation workflow."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/repo")
    # The user's own agent transcripts, read at run time and never committed.
    sessions_root: Path = Field(default_factory=claude_code_sessions_root)
    questions_file: str = "gold.jsonl"
    # A session transcript is long; keep enough of it to hold the decision.
    max_chars_per_document: int = 40000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_repo"
    session_prefix: str = "eval-repo"
    output_dir: Path = Path("evaluation/results/repo")
