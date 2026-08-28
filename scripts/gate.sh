#!/usr/bin/env bash
# The pre-push gate. Nothing reaches origin without passing every check here.
# Whole-package, not changed-files: file-scoped checking is exactly how five
# mypy errors and 32 ruff errors sat on main unseen for weeks. Runs in the
# workspace container — the host virtualenv cannot even collect the suite.
# Install with `pre-commit install --hook-type pre-push`.
#
# | Check                          | Where       |
# | ------------------------------ | ----------- |
# | ruff check + format            | local + CI  |
# | mypy                           | local + CI  |
# | import-linter                  | local + CI  |
# | bandit                         | local + CI  |
# | pytest + coverage floor        | local + CI  |
# | dependency audit               | local + CI  |
# | dependency drift               | local + CI  |
# | diff coverage                  | CI only     |
# | workflow security lint         | CI only     |
# | workflow correctness lint      | CI only     |
# (diff coverage compares against the pull request's base branch; a
# pre-push run on a local worktree has no meaningful base ref. The two
# workflow linters read only .github/workflows/*.yml, not this workspace's
# code, and neither is installed in the workspace container.)
set -uo pipefail

CONTAINER="${GRAPHKNOWS_CONTAINER:-graphknows-workspace}"
# Ratchet: env may RAISE the floor, never lower it (no `FLOOR=0 git push` bypass).
COVERAGE_FLOOR_MIN=65
COVERAGE_FLOOR="${GRAPHKNOWS_COVERAGE_FLOOR:-$COVERAGE_FLOOR_MIN}"
if [ "$COVERAGE_FLOOR" -lt "$COVERAGE_FLOOR_MIN" ] 2>/dev/null; then
  echo "gate: ignoring GRAPHKNOWS_COVERAGE_FLOOR=$COVERAGE_FLOOR - floor may be raised, never lowered." >&2
  COVERAGE_FLOOR=$COVERAGE_FLOOR_MIN
fi
AUDIT_IGNORE_FILE=".pip-audit-ignore"  # shared with .github/workflows/ci.yml

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

FAILED=()
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
record() { [ "$1" -eq 0 ] || FAILED+=("$2"); }

if ! command -v docker >/dev/null 2>&1; then
  echo "gate: docker not found. The suite cannot run on the host (incomplete venv)." >&2
  exit 1
fi
if ! docker exec "$CONTAINER" true >/dev/null 2>&1; then
  echo "gate: container '$CONTAINER' is not running. Start it: docker compose up -d" >&2
  exit 1
fi

# The shared container mounts ONE tree at /app. Pushing from a git worktree — one
# per issue, which is how this repo works — used to check that mounted tree and
# report green about code nobody was pushing. Detected by marker rather than by
# comparing paths: /app's mount source is a host path in the daemon's spelling,
# REPO_ROOT is in the shell's, and normalising between them is a portability bug
# waiting to happen. Asking the container whether it can see a file we just
# created answers the real question directly.
GATE_MARKER=".gate-tree-$$"
: >"$REPO_ROOT/$GATE_MARKER"
docker exec "$CONTAINER" test -f "/app/$GATE_MARKER" >/dev/null 2>&1
CONTAINER_HAS_THIS_TREE=$?
rm -f "$REPO_ROOT/$GATE_MARKER"

if [ "$CONTAINER_HAS_THIS_TREE" -eq 0 ]; then
  dex() { docker exec -e RUFF_CACHE_DIR=/tmp/ruffcache "$CONTAINER" sh -c "cd /app && $*"; }
else
  # Same image, so the installed venv is identical; same network, so anything
  # the suite reaches by service name still resolves.
  GATE_IMAGE="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}')"
  GATE_NET="$(docker inspect "$CONTAINER" \
    --format '{{range $net, $_ := .NetworkSettings.Networks}}{{$net}} {{end}}' | awk '{print $1}')"
  # /opt/models (NLTK_DATA, HF_HOME) is a named volume, not baked into the
  # image (see docker-compose.yaml) — an ephemeral container that skips it
  # starts with an empty corpus dir and silently degrades every WordNet-backed
  # check. Reuse whatever volume the running container mounts there instead of
  # hardcoding its compose-generated name.
  GATE_MODELS_VOL="$(docker inspect "$CONTAINER" \
    --format '{{range .Mounts}}{{if eq .Destination "/opt/models"}}{{.Name}}{{end}}{{end}}')"
  # cygpath: Docker Desktop wants a Windows path, this shell speaks MSYS.
  GATE_HOST_ROOT="$REPO_ROOT"
  if command -v cygpath >/dev/null 2>&1; then
    GATE_HOST_ROOT="$(cygpath -w "$REPO_ROOT")"
  fi
  # The ephemeral runner must inherit the reference container's volumes and
  # environment, not just its image. `model_cache:/opt/models` backs HF_HOME and
  # NLTK_DATA; without it every WordNet/embedding test ERRORs on a cold lookup
  # and anything that falls back to downloading weights runs until the caller's
  # timeout kills it. `--volumes-from` also carries that container's own `.:/app`,
  # which is the wrong tree — the explicit `-v` below overrides it at the same
  # target, which is the documented precedence and is asserted by the marker
  # check that got us here.
  # Outside the repo tree on purpose: this file is the container's whole
  # environment, API keys included, and a gate killed by a caller's timeout does
  # not always run its trap. In $TMPDIR a survivor is inert; in the worktree it
  # is one `git add -A` away from being published.
  GATE_ENV_FILE="$(mktemp)"
  chmod 600 "$GATE_ENV_FILE"
  docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' >"$GATE_ENV_FILE"
  GATE_ENV_ARG="$GATE_ENV_FILE"
  if command -v cygpath >/dev/null 2>&1; then
    GATE_ENV_ARG="$(cygpath -w "$GATE_ENV_FILE")"
  fi
  # A killed gate must not leave its container running. `--rm` fires when the
  # process exits normally; a caller timeout SIGKILLs the client and the
  # container keeps the CPU. The runbook then tells the next agent not to retry
  # -- correctly, because it would starve against the survivor -- so one orphan
  # costs the whole car. Naming it makes it killable by the trap.
  GATE_RUNNER="gk-gate-$$"
  trap 'rm -f "$GATE_ENV_FILE"; docker rm -f "$GATE_RUNNER" >/dev/null 2>&1' EXIT INT TERM
  echo "gate: '$CONTAINER' mounts a different tree - checking $REPO_ROOT instead." >&2
  dex() {
    MSYS_NO_PATHCONV=1 docker run --rm --name "$GATE_RUNNER" \
      --volumes-from "$CONTAINER" --env-file "$GATE_ENV_ARG" \
      -e RUFF_CACHE_DIR=/tmp/ruffcache \
      ${GATE_NET:+--network "$GATE_NET"} \
      -v "$GATE_HOST_ROOT:/app" -w /app \
      ${GATE_MODELS_VOL:+-v "$GATE_MODELS_VOL:/opt/models"} \
      "$GATE_IMAGE" sh -c "cd /app && $*"
  }
fi

step "ruff (check + format)"
dex "ruff check --no-fix graphknows evaluation tests"; record $? "ruff check"
dex "ruff format --check graphknows evaluation tests"; record $? "ruff format"

step "mypy (whole package)"
dex "mypy graphknows --strict --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores"
record $? "mypy"

step "import-linter (module boundaries)"
dex "lint-imports"; record $? "import-linter"

step "bandit (whole package)"
dex "bandit -r graphknows --skip B101,B104,B105,B110,B112"; record $? "bandit"

step "pytest (unit) + coverage floor ${COVERAGE_FLOOR}%"
dex "python -m pytest -q -m 'not integration' --cov=graphknows \
     --cov-report=xml:coverage.xml --cov-report=term:skip-covered --cov-fail-under=${COVERAGE_FLOOR}"
record $? "pytest/coverage"

step "dependency audit (new advisories only)"
if [ ! -f "$AUDIT_IGNORE_FILE" ]; then
  echo "gate: $AUDIT_IGNORE_FILE is missing - the ledger cannot be read." >&2
  FAILED+=("dependency audit (ledger missing)")
else
  IGNORE_ARGS="$(grep -oE '^(PYSEC|GHSA)[A-Za-z0-9-]+' "$AUDIT_IGNORE_FILE" | sed 's/^/--ignore-vuln /' | tr '\n' ' ')"
  dex "uv export --no-dev --no-editable --no-hashes --format requirements-txt > /tmp/req-audit.txt \
       && uvx pip-audit -r /tmp/req-audit.txt --progress-spinner off $IGNORE_ARGS"
  record $? "dependency audit"
fi

step "dependency drift (report-only)"
if dex "command -v deptry >/dev/null 2>&1"; then
  dex "deptry graphknows" || echo "gate: deptry findings above are report-only for now - not failing the push."
else
  echo "gate: deptry is not in the workspace image. Rebuild it: docker compose build && docker compose up -d" >&2
fi

if [ ${#FAILED[@]} -ne 0 ]; then
  printf '\n\033[31mgate: FAILED\033[0m - %s\n' "$(IFS=', '; echo "${FAILED[*]}")" >&2
  echo "Fix what the gate found. Do not weaken it to make it pass." >&2
  exit 1
fi
printf '\n\033[32mgate: passed\033[0m\n'
