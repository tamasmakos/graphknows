# ALWAYS COMMIT WITH EXTREMELY SHORT COMMIT MESSAGES AND NEVER ADD CLAUDE AS AUTHOR


# graphknows — agent contract

Agentic memory: knowledge-graph ingestion + hybrid retrieval. Package in
`graphknows/`, tests in `tests/`, benchmarks in `evaluation/`.

## Run Python in the container, never on the host

The host virtualenv is incomplete — the suite does not even collect there. Use
`docker exec graphknows-workspace ...` for pytest, mypy and `lint-imports`; a
host-run green result proves nothing. (Stdlib-only tooling that never imports
`graphknows` — the `.claude/hooks/` guard, git, docker — is not an exception,
it simply isn't covered by this rule.)

`scripts/preflight.py` hard-fails when the image's dependencies are stale
against the mounted source. If it fires run `docker compose build`; never set
`GRAPHKNOWS_SKIP_PREFLIGHT`.

## A degraded feature is a bug, not a fallback

The recurring defect here is code that runs, passes, and does nothing: a BM25
channel that silently scored with a `tf` fallback for months, a reranker that
was a no-op because its dependency was absent, a `verifier_score` computed and
discarded. All were green.

- An `except ImportError` in a core path raises `BrokenInstallError`. Only a
  genuinely optional path may degrade, and it raises `MissingExtraError` naming
  the extra.
- Never `except: pass` / `except: continue` (`S110`/`S112` are enforced). If a
  soft-fail is right, **log it** — an unlogged swallow makes "not configured
  yet" and "broken" indistinguishable from outside.
- Output is not evidence of work. Assert on ordering and values, never on
  non-emptiness. See `tests/retrieval/test_channel_liveness.py`.

## Gates

`pre-commit` (~5s, changed files): ruff check + format, mypy, bandit, Sonar.
`pre-push` (~4 min, `make gate`): full suite, whole-package mypy,
`lint-imports`, coverage floor 65%, secret scan, dependency audit, Sonar.
`make hooks` installs both, once per clone. Boundaries live in `.importlinter`
— one checker, not two.

The SonarCloud verdict is reported, not enforced: `GRAPHKNOWS_SONAR_BLOCKING`
defaults to `0`. It cannot be armed until the project's manual New Code Period
baseline is cleared (Project Settings → New Code → switch off "Specific
analysis"); until then every scanner run 422s before uploading and
`new_coverage` reads 0.0.

**Never weaken a gate to make it pass.** No `# type: ignore` without a named
reason, no extending `disable_error_code`, no lowering a threshold, no
`per-file-ignores` to silence a finding. The gate configs are blocked by
`.claude/hooks/gate_guard.py`, and `--no-verify` is denied, for the same reason:
if a gate is wrong, say so and let the human change it. Existing exemptions are
not a precedent — each carries the measurement behind it, which is what makes an
exemption reviewable rather than a silenced defect.

## Conventions

- Layers (high → low): `cli|server|integrations` → `memory` →
  `ingestion|retrieval` → `channels` → `ontology` → `ranking|topics` →
  `storage` → `models|settings|llm` → `exceptions`. A CLI is a composition root
  and lives in `graphknows/cli/`, never as a `__main__.py` in a lower layer.
- `graphknows/utils/framenet-tools` is a vendored checkout: excluded from lint
  and typing, do not reformat it.
- Two modes: `llm_free` (default, no API keys) and `llm_assisted` (the
  `assisted` extra). Anything importing `dspy`/`rdflib` is mode-gated.
- Judge noise is ±0.02. Never conclude a change helped from one run, and never
  A/B with `--no-ingest` across a mutated namespace.
