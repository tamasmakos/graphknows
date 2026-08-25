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
# Armed. Phase 2 is over: it existed because no analysis had ever uploaded coverage,
# so the verdict was meaningless, and then because main's new code definition pointed
# at a deleted analysis and every analysis of main died with a 422. The project has
# been reset and main analyses again, so the verdict now means something.
#
# Unlike COVERAGE_FLOOR above this is NOT ratcheted, so `GRAPHKNOWS_SONAR_BLOCKING=0
# git push` is a real bypass. Deliberate: Sonar is a network dependency on someone
# else's uptime, and a gate that cannot be passed when SonarCloud is down is a gate
# people uninstall. A floor you own is different from a service you do not.
SONAR_BLOCKING="${GRAPHKNOWS_SONAR_BLOCKING:-1}"

# The known-vulnerability ledger lives in .pip-audit-ignore, read by this gate
# AND by .github/workflows/ci.yml. It used to be a KNOWN_VULNS array here, which
# meant CI and the pre-push gate each carried their own copy of the same list —
# two ledgers that drift are worse than one that is occasionally stale.
AUDIT_IGNORE_FILE=".pip-audit-ignore"

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

# The shared container mounts ONE tree at /app. Pushing from a git worktree — one
# per issue, which is how this repo works — used to check that mounted tree and
# report green about code nobody was pushing. Silently validating the wrong tree
# is worse than the "silently skipped" failure the guard above refuses.
#
# Detected by marker rather than by comparing paths: /app's mount source is a host
# path in the daemon's spelling, REPO_ROOT is in the shell's, and normalising
# between them is a portability bug waiting to happen. Asking the container
# whether it can see a file we just created answers the real question directly.
GATE_MARKER=".gate-tree-$$"
: >"$REPO_ROOT/$GATE_MARKER"
docker exec "$CONTAINER" test -f "/app/$GATE_MARKER" >/dev/null 2>&1
CONTAINER_HAS_THIS_TREE=$?
rm -f "$REPO_ROOT/$GATE_MARKER"

if [ "$CONTAINER_HAS_THIS_TREE" -eq 0 ]; then
  dex() { docker exec -e RUFF_CACHE_DIR=/tmp/ruffcache "$CONTAINER" sh -c "cd /app && $*"; }
else
  # Same image, so the baked models and the installed venv are identical; same
  # network, so anything the suite reaches by service name still resolves.
  GATE_IMAGE="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}')"
  GATE_NET="$(docker inspect "$CONTAINER" \
    --format '{{range $net, $_ := .NetworkSettings.Networks}}{{$net}} {{end}}' | awk '{print $1}')"
  # cygpath: Docker Desktop wants a Windows path, this shell speaks MSYS.
  GATE_HOST_ROOT="$REPO_ROOT"
  if command -v cygpath >/dev/null 2>&1; then
    GATE_HOST_ROOT="$(cygpath -w "$REPO_ROOT")"
  fi
  echo "gate: '$CONTAINER' mounts a different tree - checking $REPO_ROOT instead." >&2
  dex() {
    MSYS_NO_PATHCONV=1 docker run --rm -e RUFF_CACHE_DIR=/tmp/ruffcache \
      ${GATE_NET:+--network "$GATE_NET"} \
      -v "$GATE_HOST_ROOT:/app" -w /app "$GATE_IMAGE" sh -c "cd /app && $*"
  }
fi

# -- 1. Lint + format, whole repo --------------------------------------------
step "ruff (check + format)"
dex "ruff check --no-fix --config ruff.toml graphknows evaluation tests"
record $? "ruff check"
dex "ruff format --check --config ruff.toml graphknows evaluation tests"
record $? "ruff format"

# -- 2. Types, whole package -------------------------------------------------
step "mypy (whole package)"
# No --exclude: CI checks the whole package, and a path the gate skips is a path
# that lands red on the runner instead of here.
dex "mypy graphknows --strict --ignore-missing-imports \
     --allow-untyped-decorators --no-warn-unused-ignores"
record $? "mypy"

# -- 3. Architecture ---------------------------------------------------------
step "import-linter (module boundaries)"
dex "lint-imports"
record $? "import-linter"

# -- 3b. Static security -----------------------------------------------------
# The pre-commit hook runs bandit on changed files only, so a finding introduced
# before the rule existed never gets seen. CI runs the whole package; so must the
# gate, with the identical skip list.
step "bandit (whole package)"
dex "bandit -r graphknows --skip B101,B104,B105,B110,B112"
record $? "bandit"

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
# First token of each non-comment line is the advisory id; the rest is the fix
# note. Reported like the missing publisher below rather than aborting the whole
# gate, and never audited with an empty ignore list - that would drown the real
# signal in known noise.
if [ ! -f "$AUDIT_IGNORE_FILE" ]; then
  echo "gate: $AUDIT_IGNORE_FILE is missing - the ledger cannot be read." >&2
  FAILED+=("dependency audit (ledger missing)")
else
  IGNORE_ARGS="$(grep -oE '^(PYSEC|GHSA)[A-Za-z0-9-]+' "$AUDIT_IGNORE_FILE" | sed 's/^/--ignore-vuln /' | tr '\n' ' ')"
  dex "uv export --no-dev --no-editable --no-hashes --format requirements-txt > /tmp/req-audit.txt \
       && uvx pip-audit -r /tmp/req-audit.txt --progress-spinner off $IGNORE_ARGS"
  record $? "dependency audit"
fi

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
