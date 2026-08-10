#!/usr/bin/env bash
# The pre-push gate. Nothing reaches origin without passing every check here.
#
# Split of responsibilities:
#   pre-commit (~5s, changed files)  ruff, mypy, bandit, sonar issue scan
#   pre-push   (this, ~3-4 min)      full suite, whole-package types, layers,
#                                    coverage, secrets, dependency audit, Sonar
#
# Whole-package, not changed-files: file-scoped checking is exactly how five
# mypy errors and 32 ruff errors sat on main unseen for weeks.
#
# Everything Python runs in the workspace container. The host virtualenv cannot
# even collect the suite, so a host run would be a green light that means
# nothing. If the container is missing this script FAILS — it never degrades to
# "skipped", which is the failure mode the whole gate exists to prevent.
#
# Install with scripts/install-hooks.sh. Bypass is `--no-verify`, available to a
# human at their own terminal and denied to agents by .claude/hooks/gate_guard.py.
set -uo pipefail

CONTAINER="${GRAPHKNOWS_CONTAINER:-graphknows-workspace}"
# The floor is a ratchet, so the environment may only RAISE it. Reading it
# straight from the env made `GRAPHKNOWS_COVERAGE_FLOOR=0 git push` a one-word
# bypass, which is the env-var escape hatch this design deliberately omits.
COVERAGE_FLOOR_MIN=65
COVERAGE_FLOOR="${GRAPHKNOWS_COVERAGE_FLOOR:-$COVERAGE_FLOOR_MIN}"
if [ "$COVERAGE_FLOOR" -lt "$COVERAGE_FLOOR_MIN" ] 2>/dev/null; then
  echo "gate: ignoring GRAPHKNOWS_COVERAGE_FLOOR=$COVERAGE_FLOOR - the floor" >&2
  echo "      may be raised but never lowered. Using $COVERAGE_FLOOR_MIN." >&2
  COVERAGE_FLOOR=$COVERAGE_FLOOR_MIN
fi
# Phase 2 uploads the Sonar analysis without gating on the verdict, so the real
# new_coverage reading is visible before it binds. Flip to 1 to arm it.
SONAR_BLOCKING="${GRAPHKNOWS_SONAR_BLOCKING:-0}"

# Vulnerabilities present when this gate was written (26 Jul 2026). Listed so
# that NEW ones fail the push while known ones do not block every commit.
# This is a debt ledger, not a dismissal — each line names its fix version.
#
#   click             8.3.2   -> 8.3.3      PYSEC-2026-2132
#   cryptography     46.0.7   -> 48.0.1     GHSA-537c-gmf6-5ccf
#   mcp              1.27.0   -> 1.28.1     PYSEC-2026-3481/3482/3483
#   pydantic-settings 2.13.1  -> 2.14.2     GHSA-4xgf-cpjx-pc3j
#   python-multipart 0.0.29   -> 0.0.31     PYSEC-2026-3036/3037/3040
#   setuptools       81.0.0   -> 83.0.0     PYSEC-2026-3447
#   starlette         1.1.0   -> 1.3.1      PYSEC-2026-248/249
#   torch            2.12.0   -> 2.13.0     PYSEC-2025-194     (image rebuild)
#   transformers     4.57.6   -> 5.x/none   PYSEC-2025-217, PYSEC-2026-2288/2289/2290
#
# The first seven are ordinary bumps. torch and transformers need an image
# rebuild and an eval re-run, so they are deliberately deferred rather than
# bundled into a gate change.
KNOWN_VULNS=(
  PYSEC-2026-2132 GHSA-537c-gmf6-5ccf
  PYSEC-2026-3481 PYSEC-2026-3482 PYSEC-2026-3483
  GHSA-4xgf-cpjx-pc3j
  PYSEC-2026-3036 PYSEC-2026-3037 PYSEC-2026-3040
  PYSEC-2026-3447
  PYSEC-2026-248 PYSEC-2026-249
  PYSEC-2025-194
  PYSEC-2025-217 PYSEC-2026-2288 PYSEC-2026-2289 PYSEC-2026-2290
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

FAILED=()
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
record() { [ "$1" -eq 0 ] || FAILED+=("$2"); }

# -- Preconditions -----------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "gate: docker not found. The suite cannot run on the host (incomplete venv)." >&2
  exit 1
fi
if ! docker exec "$CONTAINER" true >/dev/null 2>&1; then
  cat >&2 <<EOF
gate: container '$CONTAINER' is not running.

  Start it:   docker compose up -d
  Then push again.

Refusing to skip the gate - a check that silently does not run is worse than
no check at all.
EOF
  exit 1
fi

dex() { docker exec -e RUFF_CACHE_DIR=/tmp/ruffcache "$CONTAINER" sh -c "cd /app && $*"; }

# -- 1. Lint + format, whole repo --------------------------------------------
step "ruff (check + format)"
dex "ruff check --no-fix --config ruff.toml graphknows evaluation tests"
record $? "ruff check"
dex "ruff format --check --config ruff.toml graphknows evaluation tests"
record $? "ruff format"

# -- 2. Types, whole package -------------------------------------------------
step "mypy (whole package)"
dex "mypy graphknows --strict --ignore-missing-imports \
     --allow-untyped-decorators --no-warn-unused-ignores"
record $? "mypy"

# -- 3. Architecture ---------------------------------------------------------
step "import-linter (module boundaries)"
dex "lint-imports"
record $? "import-linter"

# -- 4. Tests + coverage floor -----------------------------------------------
# The floor is a ratchet against backsliding, not a target. New-code coverage is
# Sonar's new_coverage condition, which is diff coverage and the one that bites.
step "pytest (unit) + coverage floor ${COVERAGE_FLOOR}%"
dex "python -m pytest -q -m 'not integration' \
     --cov=graphknows --cov-report=xml:coverage.xml --cov-report=term:skip-covered \
     --cov-fail-under=${COVERAGE_FLOOR}"
record $? "pytest/coverage"

# -- 5. Secrets --------------------------------------------------------------
# Replaces the gitleaks stage the deleted Jenkinsfile never successfully ran.
# The SonarQube CLI is already installed and authenticated, so this costs no new
# dependency. Scoped to TRACKED files: scanning the worktree hits .venv and the
# baked models and times out at 30s, and an untracked file cannot leak anyway.
step "secret scan (tracked files)"
if command -v sonar >/dev/null 2>&1; then
  # `xargs` splits 300+ paths across several invocations, so a plain `$?` reports
  # only the LAST batch and a secret found in the first one is masked by a clean
  # second one. -P1 keeps ordering and `xargs` returns non-zero if ANY child did.
  git ls-files -z | xargs -0 -P1 sonar analyze secrets
  record $? "secret scan"
else
  echo "gate: sonar CLI not found - install it or the secret scan cannot run." >&2
  FAILED+=("secret scan (tool missing)")
fi

# -- 6. Dependency audit -----------------------------------------------------
# NOT `sonar analyze dependency-risks`: Software Composition Analysis is not
# enabled on this SonarCloud plan, so that command fails for a reason unrelated
# to the code. pip-audit gives the same signal for free.
step "dependency audit (new advisories only)"
IGNORE_ARGS=""
for v in "${KNOWN_VULNS[@]}"; do IGNORE_ARGS="$IGNORE_ARGS --ignore-vuln $v"; done
dex "uv export --no-dev --no-editable --no-hashes --format requirements-txt > /tmp/req-audit.txt \
     && uvx pip-audit -r /tmp/req-audit.txt --progress-spinner off $IGNORE_ARGS"
record $? "dependency audit"

# -- 7. SonarCloud -----------------------------------------------------------
step "SonarCloud analysis (blocking=${SONAR_BLOCKING})"
# No `if [ -f ... ]` guard: a missing publisher used to make this stage vanish
# with no output and no failure, which is the same silent skip the header of this
# file forbids. An absent script is a broken gate, so it is reported as one.
if [ ! -f scripts/sonar-publish.sh ]; then
  echo "gate: scripts/sonar-publish.sh is missing - the Sonar stage cannot run." >&2
  FAILED+=("SonarCloud analysis (script missing)")
else
  bash scripts/sonar-publish.sh
  rc=$?
  if [ "$SONAR_BLOCKING" = "1" ]; then
    record $rc "SonarCloud quality gate"
  elif [ $rc -ne 0 ]; then
    echo "gate: Sonar analysis reported failure - NOT blocking this push yet." >&2
  fi
fi

# -- Verdict -----------------------------------------------------------------
if [ ${#FAILED[@]} -ne 0 ]; then
  printf '\n\033[31mgate: FAILED\033[0m - %s\n' "$(IFS=', '; echo "${FAILED[*]}")" >&2
  echo "Fix what the gate found. Do not weaken it to make it pass." >&2
  exit 1
fi
printf '\n\033[32mgate: passed\033[0m\n'
