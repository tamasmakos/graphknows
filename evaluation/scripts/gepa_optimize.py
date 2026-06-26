"""GEPA prompt optimization for the LoCoMo answer generator — leakage-safe.

Train/val on conv-30; evaluate on held-out conv-26 (different conversation, no
question overlap). The few-shot demos use invented people, so no benchmark facts
leak. GEPA never sees conv-26 during compile.

  MODE=llm_free python -m evaluation.scripts.gepa_optimize --smoke   # wiring check
  MODE=llm_free python -m evaluation.scripts.gepa_optimize --budget 150
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv("/app/.env", override=True)

from evaluation.dataset import build_documents, first_string, read_rows  # noqa: E402
from evaluation.generation.signatures import LoCoMoAnswerGeneration, demos  # noqa: E402
from evaluation.scoring.metrics import normalize_answer, token_f1  # noqa: E402
from graphknows.config import GraphKnowsSettings  # noqa: E402
from graphknows.config.llm import _get_openrouter_api_key, normalize_openrouter_model  # noqa: E402
from graphknows.runtime import GraphKnowsRuntime  # noqa: E402

OUT = Path("evaluation/results/gepa")
CACHE = OUT / "examples.json"
TRAIN_DLG = "conv-30"   # optimization conversation
TEST_DLG = "conv-26"    # held-out evaluation conversation
TEST_SRC = Path("evaluation/results/patched-sample/patched-llm_free.json")


def _env_model() -> str:
    for line in Path(".env").read_text().splitlines():
        if line.strip().startswith("LLM_MODEL="):
            return line.split("=", 1)[1].strip()


def _ctx_from_hits(hits: list[dict], k: int = 25) -> str:
    lines, seen = [], set()
    for h in hits[:k]:
        t = (h.get("text") or "").strip()
        if t and t not in seen:
            seen.add(t)
            lines.append(f"[{len(lines)+1}] {t}")
    return "\n\n".join(lines)[:24_000]


# ── Build leakage-safe datasets ────────────────────────────────────────────────
async def _build_train_examples() -> list[dict]:
    """Ingest conv-30 and retrieve a real top-25 context for each of its questions."""
    rows = read_rows(Path("evaluation/data/locomo/questions.jsonl"))
    dlg_rows = [r for r in rows if first_string(r, ("id",), "").startswith(TRAIN_DLG)]
    rt = GraphKnowsRuntime(GraphKnowsSettings())
    sid = f"gepa-{TRAIN_DLG}"
    docs = build_documents(dlg_rows[0], max_documents=50, max_chars=8000)
    await rt.purge_memory(session_id="", scope="ltm")
    await rt.purge_memory(session_id=sid, scope="both")
    print(f"[gepa] ingesting {len(docs)} {TRAIN_DLG} docs …", flush=True)
    for i, d in enumerate(docs):
        await rt.ingest_memory(text=d.text, session_id=sid, title=d.title or f"d{i}",
                               metadata={"anchor_date": d.anchor_date})
    await rt.flush_memory(sid)
    out = []
    for r in dlg_rows:
        q = first_string(r, ("question",))
        gold = str(r.get("answer") or "")
        if not q or not gold:
            continue
        res = await rt.recall_memory(q, session_id=sid, top_k=25, scope="ltm")
        out.append({"question": q, "retrieved_context": _ctx_from_hits(res.get("hits", [])),
                    "answer": gold})
    await rt.close()
    return out


def _load_test_examples() -> list[dict]:
    cases = json.load(open(TEST_SRC))["cases"]
    out = []
    for c in cases:
        mc = c.get("memory_context") or []
        ctx = "\n\n".join(f"[{i+1}] {(it.get('text') if isinstance(it, dict) else str(it))}"
                          for i, it in enumerate(mc))[:24_000]
        out.append({"question": c["question"], "retrieved_context": ctx, "answer": c["gold"]})
    return out


def build_or_load() -> tuple[list[dict], list[dict]]:
    if CACHE.exists():
        d = json.loads(CACHE.read_text())
        return d["train"], d["test"]
    OUT.mkdir(parents=True, exist_ok=True)
    train = asyncio.run(_build_train_examples())
    test = _load_test_examples()
    CACHE.write_text(json.dumps({"train": train, "test": test}))
    print(f"[gepa] cached {len(train)} train (conv-30) + {len(test)} test (conv-26)")
    return train, test


# ── Program + feedback metric ──────────────────────────────────────────────────
class GenProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought(LoCoMoAnswerGeneration)
        self.gen.demos = demos()

    def forward(self, question, retrieved_context):
        return self.gen(question=question, retrieved_context=retrieved_context)


# Function words to ignore when scoring, so GEPA cannot "improve" the metric by
# instructing the model to copy the gold's articles/conjunctions verbatim (which
# token-F1 would reward). Scoring on content words keeps optimization semantic.
_STOP = {
    "a", "an", "the", "and", "or", "but", "by", "of", "to", "in", "on", "at", "for",
    "with", "own", "is", "was", "are", "were", "be", "been", "her", "his", "she",
    "he", "it", "its", "they", "them", "their", "as", "that", "this", "these",
    "those", "from", "than", "then", "so", "i", "you", "we", "my", "your", "our",
}


def _content(s: str) -> list[str]:
    return [t for t in normalize_answer(s).split() if t not in _STOP]


def content_f1(gold: str, pred: str) -> float:
    from collections import Counter

    gt, pt = _content(gold), _content(pred)
    if not gt or not pt:
        return 0.0
    common = Counter(gt) & Counter(pt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)


class GepaJudge(dspy.Signature):
    """Judge a predicted answer against the gold answer for a question about a long
    two-person conversation, and explain the gap.

    correctness: 1.0 when the prediction conveys ALL the gold's facts and qualifiers
    (paraphrase/synonyms fine); lower as facts are missing, wrong, or the prediction
    abstains ('Unknown') when the answer is actually derivable from the gold.
    feedback: one or two sentences of SPECIFIC, actionable guidance for the answer
    writer — name the missing fact, or flag a wrong / over-abstained / over-verbose
    answer. Describe what to fix; do not quote the gold verbatim."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    correctness: float = dspy.OutputField(desc="0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="specific, actionable feedback")


# Set in main(): the semantic judge used as the GEPA metric (real task objective +
# rich feedback, per the DSPy GEPA guidance). Falls back to content-F1 if unset/errors.
_JUDGE: dict = {"predictor": None, "lm": None}


def _heuristic_feedback(g: str, p: str, score: float) -> str:
    gt, pt = set(_content(g)), set(_content(p))
    fb = []
    if p.strip().lower().startswith("unknown") and g.strip().lower() not in ("unknown", ""):
        fb.append("OVER-ABSTENTION: answered 'Unknown' though the answer is supported by the "
                  "context; extract it.")
    missing = sorted(gt - pt)
    if missing and score < 1.0:
        fb.append(f"INCOMPLETE: missing {missing[:6]}; include every gold fact/qualifier.")
    extra = sorted(pt - gt)
    if len(pt) > 2 * max(1, len(gt)) and extra:
        fb.append(f"TOO VERBOSE: drop {extra[:6]}; give only the bare answer.")
    return " ".join(fb) or f"Partial match (F1={score:.2f}); match the gold's content and brevity."


def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    g = str(getattr(gold, "answer", "") or "")
    p = str(getattr(pred, "answer", "") or "")
    if not p:
        return dspy.Prediction(score=0.0,
                               feedback="Empty answer — produce a concrete answer from the passages.")
    if _JUDGE["predictor"] is not None:
        try:
            with dspy.settings.context(lm=_JUDGE["lm"]):
                j = _JUDGE["predictor"](question=str(getattr(gold, "question", "")),
                                        gold_answer=g, predicted_answer=p)
            score = max(0.0, min(1.0, float(str(getattr(j, "correctness", "0")).strip() or 0)))
            fb = str(getattr(j, "feedback", "") or "").strip() or _heuristic_feedback(g, p, score)
            return dspy.Prediction(score=score, feedback=fb)
        except Exception:
            pass  # any judge failure → lexical fallback so a candidate is never lost
    score = content_f1(g, p)
    return dspy.Prediction(score=score, feedback=_heuristic_feedback(g, p, score))


def evaluate(program, examples, label):
    lm = dspy.settings.lm
    tot = 0.0
    for ex in examples:
        try:
            with dspy.settings.context(lm=lm):
                pred = program(question=ex.question, retrieved_context=ex.retrieved_context)
            ans = str(getattr(pred, "answer", "") or "")
        except Exception:
            ans = ""  # reasoning-model truncation etc. → scores 0, never crashes the eval
        tot += token_f1(ex.answer, ans)
    print(f"[eval:{label}] mean token-F1 = {tot/len(examples):.4f}  over {len(examples)}")
    return tot / len(examples)


# A deliberately BAD seed for the --badseed optimizer check: it forces long,
# explanatory paragraphs, which tank token-F1 against LoCoMo's terse gold answers.
# A working optimizer should discover it must be terse/extractive and recover.
_BAD_SEED = (
    "Write a thorough, multi-sentence paragraph that fully explains your reasoning "
    "and describes every relevant detail from the passages when answering the question."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--badseed", action="store_true",
                    help="start from a deliberately weak instruction to verify GEPA can improve it")
    ap.add_argument("--budget", type=int, default=150, help="max_metric_calls")
    args = ap.parse_args()

    model = normalize_openrouter_model(_env_model())
    key = _get_openrouter_api_key()
    print(f"[gepa] model (task + reflection + judge): {model}")
    task_lm = dspy.LM(model=model, api_key=key, api_base="https://openrouter.ai/api/v1",
                      temperature=0.0, max_tokens=16000, num_retries=2, cache=False)
    reflection_lm = dspy.LM(model=model, api_key=key, api_base="https://openrouter.ai/api/v1",
                            temperature=1.0, max_tokens=16000, num_retries=2, cache=False)
    # Semantic judge metric (the real objective + rich feedback), same model.
    _JUDGE["lm"] = dspy.LM(model=model, api_key=key, api_base="https://openrouter.ai/api/v1",
                           temperature=0.0, max_tokens=8000, num_retries=2, cache=False)
    _JUDGE["predictor"] = dspy.Predict(GepaJudge)
    dspy.configure(lm=task_lm)

    train_raw, test_raw = build_or_load()

    def mk(rows):
        return [dspy.Example(question=r["question"], retrieved_context=r["retrieved_context"],
                             answer=r["answer"]).with_inputs("question", "retrieved_context")
                for r in rows]

    train_ex = mk(train_raw)
    test_ex = mk(test_raw)
    # Train/val split WITHIN conv-30 (test stays conv-26, fully held out).
    n_tr = len(train_ex)
    if args.smoke:
        trainset, valset = train_ex[:8], train_ex[8:13]
        test_ex = test_ex[:8]  # held-out eval big enough to show a format fix
        budget = {"auto": "light"}  # ~6 candidates — the doc's proven minimal budget
    else:
        cut = int(n_tr * 0.7)
        trainset, valset = train_ex[:cut], train_ex[cut:]
        budget = {"max_metric_calls": args.budget}
    print(f"[gepa] train={len(trainset)} val={len(valset)} (conv-30)  test={len(test_ex)} (conv-26)  budget={budget}")

    student = GenProgram()
    if args.badseed:
        # Replace the (already-good) instruction with a weak one so we can SEE GEPA
        # improve it — the cleanest "is the optimizer actually working" check.
        student.gen.predict.signature = student.gen.predict.signature.with_instructions(_BAD_SEED)
        print(f"[gepa] BAD-SEED mode: starting instruction = {_BAD_SEED!r}")
    base = evaluate(student, test_ex, "baseline/conv-26")

    gepa = dspy.GEPA(metric=feedback_metric, reflection_lm=reflection_lm,
                     num_threads=4, track_stats=True, log_dir=str(OUT / "log"), seed=42, **budget)
    optimized = gepa.compile(student, trainset=trainset, valset=valset)

    instr = optimized.gen.predict.signature.instructions
    (OUT / "optimized_instructions.txt").write_text(instr, encoding="utf-8")
    print("\n========== OPTIMIZED INSTRUCTION ==========\n" + instr + "\n===========================================\n")

    opt = evaluate(optimized, test_ex, "optimized/conv-26")
    print(f"\n[RESULT] conv-26 held-out token-F1: baseline {base:.4f} -> optimized {opt:.4f}  "
          f"(Δ {opt-base:+.4f})")
    optimized.save(str(OUT / "optimized_program.json"))


if __name__ == "__main__":
    main()
