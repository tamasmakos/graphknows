# graphknows — agent contract

Agentic memory: knowledge-graph ingestion + hybrid retrieval. Python package in
`graphknows/`, tests in `tests/`, benchmarks in `evaluation/`.

## Run everything in the container, never on the host

The host virtualenv is incomplete — the test suite does not even collect there
(`No module named dateparser / rank_bm25 / dspy`). Every Python command goes
through the workspace container:

```
docker exec graphknows-workspace python -m pytest -q -m "not integration"
docker exec graphknows-workspace mypy graphknows --exclude "graphknows/utils" --strict --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores
docker exec graphknows-workspace lint-imports
```

A host-run green result proves nothing. `scripts/preflight.py` hard-fails when
the image's dependencies are stale relative to the mounted source — if it fires,
run `docker compose build`, do not set `GRAPHKNOWS_SKIP_PREFLIGHT`.

## A degraded feature is a bug, not a fallback

This repo's recurring defect is code that runs, passes, and does nothing: a BM25
channel that silently scored with a `tf` fallback for months, a cross-encoder
reranker that was a no-op because its dependency was absent, a `verifier_score`
computed and then discarded. All were green.

- An `except ImportError` in a core path must raise `BrokenInstallError`, not
  degrade. Only a genuinely optional path may degrade, and it raises
  `MissingExtraError` naming the extra.
- Never `except: pass` / `except: continue`. `S110`/`S112` are enforced. If a
  soft-fail is right, **log it** — an unlogged swallow makes "not configured yet"
  and "broken" indistinguishable from outside.
- A retrieval channel that produces output is not evidence it works. Assert on
  ordering and values, not on non-emptiness.

## Gates

`pre-commit` (~5s, on changed files): ruff check + format, mypy, bandit, Sonar.
`pre-push` (~2-4 min): full unit suite, whole-package mypy, `lint-imports`,
gitleaks, pip-audit, coverage floor, SonarCloud quality gate.

`make ci` runs the local equivalent. Architecture boundaries live in
`.importlinter` — one checker, not two.

**Never weaken a gate to make it pass.** Do not add `# type: ignore` without a
named reason, do not extend `disable_error_code`, do not lower a coverage
threshold, do not add a `per-file-ignores` entry to silence a finding. Editing
the gate configs is blocked by `.claude/hooks/gate_guard.py`; if a gate is wrong,
say so and let the human change it. `--no-verify` is denied for the same reason.

## Conventions

- Layers (high → low): `cli|server|integrations` → `memory` →
  `ingestion|retrieval` → `channels` → `ontology` → `ranking|topics` → `storage`
  → `models|settings|llm` → `exceptions`. A CLI is a composition root and lives
  in `graphknows/cli/`, never as a `__main__.py` inside a lower layer.
- `graphknows/utils/framenet-tools` is a vendored third-party checkout: excluded
  from lint and typing, do not reformat it.
- Two extraction modes, `llm_free` (default, no API keys) and `llm_assisted`
  (the `assisted` extra). Anything importing `dspy`/`rdflib` is mode-gated.
- Evaluation judge noise is ±0.02. Never conclude a change helped from a single
  run, and never A/B with `--no-ingest` across a mutated namespace.
