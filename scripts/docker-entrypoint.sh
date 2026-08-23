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

exec "$@"
