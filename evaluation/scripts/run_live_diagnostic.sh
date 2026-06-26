#!/usr/bin/env bash
# Live LoCoMo diagnostic — runs BOTH methods end-to-end inside graphknows-workspace
# and re-grades every case with the two DSPy semantic metrics (SemanticF1 +
# CompleteAndGrounded), with heavy instrumentation captured to log files.
#
# Run from the HOST (the repo is bind-mounted at /app inside the container):
#
#     docker exec graphknows-workspace bash evaluation/scripts/run_live_diagnostic.sh        # 50 questions
#     docker exec graphknows-workspace bash evaluation/scripts/run_live_diagnostic.sh 10     # quick smoke
#
# All outputs land in evaluation/results/diagnostic-probe-20260625/live/ (on the
# host too, via the bind mount). Requires OPENROUTER_API_KEY + LLM_MODEL in .env
# and a running arcadedb service — i.e. the normal eval prerequisites.
set -uo pipefail

LIMIT="${1:-50}"
cd /app || { echo "FATAL: /app not found (run inside graphknows-workspace)"; exit 1; }

PY="$(command -v python || echo /opt/venv/bin/python)"
OUT="evaluation/results/diagnostic-probe-20260625/live"
mkdir -p "$OUT"

# Heavier instrumentation: surface litellm/dspy request+response + graphknows logs.
export LITELLM_LOG="${LITELLM_LOG:-INFO}"
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo " Live LoCoMo diagnostic   limit=$LIMIT   python=$PY"
echo " started: $(date -u +%FT%TZ)"
echo "=================================================================="

run_mode () {
  local mode="$1"
  local log="$OUT/run-${mode}.log"
  # Progress lines go to stderr: stdout is captured by the caller as the result
  # JSON path, so anything else on stdout would corrupt that path.
  echo >&2; echo ">>> [$mode] running python -m evaluation --mode $mode --limit $LIMIT" >&2
  local t0 t1
  t0=$(date +%s)
  # 2>&1 captures retrieval traces, LLM prompts/responses (LITELLM_LOG), retries,
  # provider 400/429 fallbacks, parse-failure abstentions, and per-case lines.
  "$PY" -m evaluation --mode "$mode" --limit "$LIMIT" >"$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  echo ">>> [$mode] exit=$rc  wall=$((t1 - t0))s  log=$log" >&2
  if [ $rc -ne 0 ]; then
    echo ">>> [$mode] FAILED — tail of log:" >&2; tail -25 "$log" >&2
  fi
  # Newest result JSON for this mode (cli.py names them eval-<mode>-<ts>-locomo.json)
  ls -t evaluation/results/locomo/eval-"${mode}"-*-locomo.json 2>/dev/null | head -1
}

FREE_JSON="$(run_mode llm_free)"
ASST_JSON="$(run_mode llm_assisted)"
echo; echo "llm_free   result: ${FREE_JSON:-<none>}"
echo      "llm_assisted result: ${ASST_JSON:-<none>}"

regrade () {
  local mode="$1" src="$2"
  [ -z "$src" ] && { echo ">>> [$mode] no result JSON; skipping regrade"; return; }
  echo; echo ">>> [$mode] re-grading with SemanticF1 + CompleteAndGrounded"
  "$PY" -m evaluation.scripts.regrade_dspy_metrics \
      --input "$src" \
      --output "$OUT/regraded-${mode}.json" 2>&1 | tee "$OUT/regrade-${mode}.log"
}

regrade llm_free "$FREE_JSON"
regrade llm_assisted "$ASST_JSON"

# Combined comparison table across all four metrics for both methods.
"$PY" - "$OUT" <<'PY'
import json, sys, glob, statistics as st, os
out = sys.argv[1]
def load(mode):
    fs = sorted(glob.glob(f"{out}/regraded-{mode}.json"), key=os.path.getmtime)
    return json.load(open(fs[-1])) if fs else None
rows = []
for mode in ("llm_free", "llm_assisted"):
    d = load(mode)
    if not d:
        rows.append((mode, None)); continue
    cs = d["cases"]
    def m(key): return round(st.mean(c["metrics"].get(key, 0.0) for c in cs), 4)
    rows.append((mode, dict(
        n=len(cs),
        token_f1=round(st.mean(c["score"] for c in cs), 4),
        judge=round(st.mean(c["metrics"].get("judge_correct", 0) for c in cs), 4),
        semantic_f1=m("semantic_f1"),
        complete_and_grounded=m("complete_and_grounded"),
        completeness=m("completeness"),
        groundedness=m("groundedness"),
        unknown=sum(c["generated_answer"].strip().lower().startswith("unknown") for c in cs),
    )))
md = ["# Live diagnostic — combined metric table\n",
      "| metric | llm_free | llm_assisted |", "|---|---:|---:|"]
keys = ["n","token_f1","judge","semantic_f1","complete_and_grounded","completeness","groundedness","unknown"]
f = dict(rows).get("llm_free") or {}
a = dict(rows).get("llm_assisted") or {}
for k in keys:
    md.append(f"| {k} | {f.get(k,'—')} | {a.get(k,'—')} |")
open(f"{out}/LIVE_COMBINED_METRICS.md","w").write("\n".join(md)+"\n")
json.dump({"llm_free":f,"llm_assisted":a}, open(f"{out}/live_combined_metrics.json","w"), indent=2)
print("\n".join(md))
PY

echo; echo "=================================================================="
echo " DONE  $(date -u +%FT%TZ)"
echo " Artifacts in: $OUT/"
echo "   run-<mode>.log            full stdout (retrieval traces, LLM I/O, retries, fallbacks)"
echo "   regraded-<mode>.json/.csv per-case token_f1 + semantic_f1 + complete_and_grounded"
echo "   LIVE_COMBINED_METRICS.md  side-by-side table for both methods, both metrics"
echo "=================================================================="
