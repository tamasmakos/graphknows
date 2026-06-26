#!/usr/bin/env bash
# Overnight LoCoMo run — BOTH methods, scored with DSPy SemanticF1 +
# CompleteAndGrounded. Uses the proven two-stage flow:
#   1. eval  (generation = LLM_MODEL/kimi-k2, the validated baseline) — token-F1
#      per case, per-dialogue + every-30-case checkpointed so a crash never loses
#      work.
#   2. regrade with a FAST scoring model (SCORE_MODEL, default gpt-4o-mini): the
#      DSPy judge need not be the generation model, and kimi-k2 is a slow reasoning
#      model (~4 min/case) that makes inline scoring infeasible overnight.
#
#   docker exec graphknows-workspace bash evaluation/scripts/run_overnight_locomo.sh [LIMIT]
#
# LIMIT defaults to 152 = the full conv-26 conversation (both modes complete in ~6h).
set -uo pipefail

LIMIT="${1:-152}"
SCORE_MODEL="${SCORE_MODEL:-openai/gpt-4o-mini}"
cd /app || { echo "FATAL: /app not found"; exit 1; }

PY="$(command -v python || echo /opt/venv/bin/python)"
OUT="evaluation/results/overnight-locomo"
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1
export LITELLM_LOG="${LITELLM_LOG:-WARNING}"

echo "=================================================================="
echo " Overnight LoCoMo   limit=$LIMIT   score_model=$SCORE_MODEL"
echo " start=$(date -u +%FT%TZ)"
echo "=================================================================="

# ── Stage 1: evaluate (generation only; fast, checkpointed) ──────────────────
run_mode () {   # echoes the result JSON path on stdout; progress to stderr
  local mode="$1"
  local log="$OUT/run-${mode}.log"
  echo ">>> [$mode] eval python -m evaluation --mode $mode --limit $LIMIT" >&2
  local t0 t1; t0=$(date +%s)
  "$PY" -m evaluation --mode "$mode" --limit "$LIMIT" >"$log" 2>&1
  local rc=$?; t1=$(date +%s)
  echo ">>> [$mode] eval exit=$rc wall=$((t1-t0))s log=$log" >&2
  ls -t evaluation/results/locomo/eval-"${mode}"-*-locomo.json 2>/dev/null | head -1
}

# llm_assisted first — it is the mode under investigation, so an overrun secures
# the priority result first (it is also independently checkpointed).
ASST_EVAL="$(run_mode llm_assisted)"
FREE_EVAL="$(run_mode llm_free)"
echo "asst eval -> ${ASST_EVAL:-<none>}"
echo "free eval -> ${FREE_EVAL:-<none>}"

# ── Stage 2: regrade with the fast DSPy scorer ───────────────────────────────
regrade () {   # echoes the regraded JSON path on stdout
  local mode="$1" src="$2"
  [ -z "$src" ] && { echo ">>> [$mode] no eval JSON; skip regrade" >&2; return; }
  local out="$OUT/regraded-${mode}.json"
  echo ">>> [$mode] regrade ($SCORE_MODEL) -> $out" >&2
  OPENROUTER_MODEL="$SCORE_MODEL" "$PY" -m evaluation.scripts.regrade_dspy_metrics \
      --input "$src" --output "$out" >"$OUT/regrade-${mode}.log" 2>&1
  echo "$out"
}

ASST_JSON="$(regrade llm_assisted "$ASST_EVAL")"
FREE_JSON="$(regrade llm_free "$FREE_EVAL")"

# ── Stage 3: side-by-side comparison ─────────────────────────────────────────
"$PY" - "$OUT" "$FREE_JSON" "$ASST_JSON" <<'PY'
import json, sys, statistics as st
out, free_p, asst_p = sys.argv[1], sys.argv[2], sys.argv[3]
def row(p):
    if not p:
        return None
    try:
        cs = json.load(open(p))["cases"]
    except Exception:
        return None
    def m(k): return round(st.mean(c["metrics"].get(k, 0.0) for c in cs), 4) if cs else 0.0
    return dict(
        n=len(cs),
        dialogues=len({c.get("dialogue_id") for c in cs}),
        token_f1=round(st.mean(c["score"] for c in cs), 4) if cs else 0.0,
        semantic_f1=m("semantic_f1"),
        complete_and_grounded=m("complete_and_grounded"),
        completeness=m("completeness"),
        groundedness=m("groundedness"),
        unknown=sum(c["generated_answer"].strip().lower().startswith("unknown") for c in cs),
    )
f, a = row(free_p), row(asst_p)
keys = ["n","dialogues","token_f1","semantic_f1","complete_and_grounded","completeness","groundedness","unknown"]
md = ["# Overnight LoCoMo — llm_free vs llm_assisted",
      "", "Scoring: DSPy SemanticF1 + CompleteAndGrounded (judge model: see run).",
      "", "| metric | llm_free | llm_assisted |", "|---|---:|---:|"]
for k in keys:
    md.append(f"| {k} | {(f or {}).get(k,'—')} | {(a or {}).get(k,'—')} |")
open(f"{out}/OVERNIGHT_COMBINED_METRICS.md","w").write("\n".join(md)+"\n")
json.dump({"llm_free": f, "llm_assisted": a}, open(f"{out}/overnight_combined_metrics.json","w"), indent=2)
print("\n".join(md))
PY

echo "=================================================================="
echo " DONE  $(date -u +%FT%TZ)   artifacts in $OUT/"
echo "=================================================================="
