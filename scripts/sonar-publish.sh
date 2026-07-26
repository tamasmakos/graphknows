#!/usr/bin/env bash
# Publish a full SonarCloud analysis, including the coverage report.
#
# This is the step nothing ever ran. The quality gate's `new_coverage` condition
# read 0.0 — and therefore ERROR — not because the code was untested but because
# no analysis ever uploaded coverage.xml. Publishing was the deleted Jenkins
# pipeline's job, and it never went green.
#
# The scanner runs from its official image rather than being installed: the
# SonarQube CLI on PATH is the client-side analyzer (issues in a change set) and
# cannot publish a project analysis.
#
# GRAPHKNOWS_SONAR_BLOCKING=1 makes the scanner wait for the server verdict and
# exit non-zero on ERROR. With 0 it uploads and returns success, so the real
# new_coverage reading is visible before it starts blocking pushes.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

SONAR_BLOCKING="${GRAPHKNOWS_SONAR_BLOCKING:-0}"
SCANNER_IMAGE="${SONAR_SCANNER_IMAGE:-sonarsource/sonar-scanner-cli:latest}"

# Token from the environment, else from .env (never echoed).
if [ -z "${SONAR_TOKEN:-}" ] && [ -f .env ]; then
  SONAR_TOKEN="$(grep -E '^SONAR_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
fi
if [ -z "${SONAR_TOKEN:-}" ]; then
  echo "sonar: SONAR_TOKEN not set and not found in .env — cannot publish." >&2
  exit 1
fi

if [ ! -f coverage.xml ]; then
  echo "sonar: coverage.xml missing. Run the test step first — publishing without" >&2
  echo "       it is what pinned new_coverage at 0.0 in the first place." >&2
  exit 1
fi

WAIT_FLAG="false"
[ "$SONAR_BLOCKING" = "1" ] && WAIT_FLAG="true"

# MSYS_NO_PATHCONV stops Git Bash from mangling the container-side path.
MSYS_NO_PATHCONV=1 docker run --rm \
  -e SONAR_HOST_URL="${SONAR_HOST_URL:-https://sonarcloud.io}" \
  -e SONAR_TOKEN="$SONAR_TOKEN" \
  -v "$REPO_ROOT:/usr/src" \
  "$SCANNER_IMAGE" \
  -Dsonar.qualitygate.wait="$WAIT_FLAG"
rc=$?

if [ $rc -ne 0 ] && [ "$WAIT_FLAG" = "true" ]; then
  echo "sonar: quality gate FAILED. See https://sonarcloud.io/project/overview?id=tamasmakos_graphknows" >&2
fi
exit $rc
