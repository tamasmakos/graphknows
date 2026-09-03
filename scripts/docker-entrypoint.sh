#!/bin/sh
# Container entrypoint: verify the image is not stale, then hand off.
#
# Wraps EVERY command — graphknows-mcp and the dev shell — so a
# dependency mismatch cannot be walked past. See scripts/preflight.py for what is
# checked and why, and GRAPHKNOWS_SKIP_PREFLIGHT=1 to bypass.
#
# `exec` replaces this shell with the real process, so signals and exit codes pass
# through unchanged (no PID-1 shell swallowing SIGTERM on `docker compose down`).
set -e

python "$(dirname "$0")/preflight.py" || exit 1

# Release image only (GRAPHKNOWS_REQUIRE_BAKED=1): fail at boot on an empty model
# cache instead of mid-ingest on an air-gapped host. The dev container leaves the
# variable unset and keeps its lazy cold start. `--present`, never `--check`: the
# deep check loads every model and costs ~5x an ingest (research.md R2). `set -e`
# above turns a failure here into a refused boot.
[ "${GRAPHKNOWS_REQUIRE_BAKED:-}" = "1" ] && python "$(dirname "$0")/bake_models.py" --present

exec "$@"
