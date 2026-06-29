#!/usr/bin/env bash
# Live LoCoMo diagnostic — runs BOTH methods end-to-end inside graphknows-workspace.
# SemanticF1 is now computed inline during each eval run (no separate re-grade step).
# A side-by-side per-category comparison table is produced by --compare.
#
# Run from the HOST (the repo is bind-mounted at /app inside the container):
#
#     docker exec graphknows-workspace bash evaluation/scripts/run_live_diagnostic.sh        # 50 questions
#     docker exec graphknows-workspace bash evaluation/scripts/run_live_diagnostic.sh 10     # quick smoke
#
# All outputs land in evaluation/results/diagnostic-probe/ (on the host too,
# via the bind mount). Requires OPENROUTER_API_KEY + LLM_MODEL in .env and a
# running arcadedb service — the normal eval prerequisites.
set -uo pipefail

LIMIT="${1:-50}"
cd /app || { echo "FATAL: /app not found (run inside graphknows-workspace)"; exit 1; }

PY="$(command -v python || echo /opt/venv/bin/python)"
TS="$(date -u +%Y%m%d%H%M%S)"
OUT="evaluation/results/diagnostic-probe-${TS}/live"
mkdir -p "$OUT"

export LITELLM_LOG="${LITELLM_LOG:-INFO}"
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo " Live LoCoMo diagnostic   limit=$LIMIT   python=$PY"
echo " started: $(date -u +%FT%TZ)"
echo "=================================================================="

LOG="$OUT/compare.log"
echo ">>> running --compare (llm_free + llm_assisted inline)  limit=$LIMIT" >&2
"$PY" -m evaluation --compare --limit "$LIMIT" 2>&1 | tee "$LOG"

echo
echo "=================================================================="
echo " DONE  $(date -u +%FT%TZ)"
echo " Artifacts in: $OUT/"
echo "   compare.log          full output including per-category SemanticF1 table"
echo "   evaluation/results/locomo/compare-*.md  persistent comparison table"
echo "=================================================================="
