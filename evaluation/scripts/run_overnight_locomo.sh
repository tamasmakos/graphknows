#!/usr/bin/env bash
# Overnight LoCoMo run — BOTH methods, scored with DSPy SemanticF1 (inline,
# single-stage). SemanticF1 is computed per-case during generation; no separate
# re-grade pass is needed.
#
# Uses --compare to run both llm_free and llm_assisted and produce a side-by-side
# per-category SemanticF1 table.
#
#   docker exec graphknows-workspace bash evaluation/scripts/run_overnight_locomo.sh [LIMIT]
#
# LIMIT defaults to 152 = the full conv-26 conversation (both modes complete in ~6h).
set -uo pipefail

LIMIT="${1:-152}"
cd /app || { echo "FATAL: /app not found"; exit 1; }

PY="$(command -v python || echo /opt/venv/bin/python)"
OUT="evaluation/results/overnight-locomo"
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1
export LITELLM_LOG="${LITELLM_LOG:-WARNING}"

echo "=================================================================="
echo " Overnight LoCoMo   limit=$LIMIT   model=$(grep LLM_MODEL .env | head -1 | cut -d= -f2)"
echo " start=$(date -u +%FT%TZ)"
echo "=================================================================="

# Run both modes with inline SemanticF1 scoring and write the comparison table.
LOG="$OUT/overnight-$(date -u +%Y%m%d%H%M%S).log"
"$PY" -m evaluation --compare --limit "$LIMIT" 2>&1 | tee "$LOG"

echo "=================================================================="
echo " DONE  $(date -u +%FT%TZ)   log=$LOG"
echo " Comparison table written to evaluation/results/locomo/compare-*.md"
echo "=================================================================="
