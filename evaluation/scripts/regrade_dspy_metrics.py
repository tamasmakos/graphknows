"""Re-grade a LoCoMo results JSON with DSPy SemanticF1 + CompleteAndGrounded.

The main evaluation already writes per-case lexical token-F1 and a binary LLM
judge. This script adds the two reference DSPy semantic metrics *after the fact*,
so any past or future run can be re-scored without re-ingesting or re-retrieving:

    python -m evaluation.scripts.regrade_dspy_metrics \
        --input  evaluation/results/locomo/eval-llm_assisted-XXlike.json \
        --output evaluation/results/diagnostic-probe/regraded-llm_assisted.json

It reads each case's ``question``, ``gold``, ``generated_answer`` and the
retrieved ``passages`` (used as the grounding context for CompleteAndGrounded),
calls the two DSPy metrics via OpenRouter (``OPENROUTER_API_KEY`` + ``LLM_MODEL``
from ``.env``), and writes:

    <output>.json   the input augmented with semantic_f1 / complete_and_grounded /
                    completeness / groundedness on each case + an aggregate block
    <output>.csv    one flat row per case for analysis

``--selftest`` validates the metric wiring fully offline with a stub LM (no
network, no API key) and exits.

NOTE: grading is an LLM operation. It will NOT work in a sandbox without network
access to OpenRouter; run it on a host where the model endpoint is reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics as st
from pathlib import Path

log = logging.getLogger(__name__)

# How many top passages to hand the groundedness check as the "retrieved context".
_CONTEXT_K = 25
_CONTEXT_CHAR_BUDGET = 16000


def _context_from_passages(case: dict) -> str:
    """Join the retrieved passage texts into a grounding context string."""
    passages = case.get("passages") or case.get("memory_context") or []
    seen: set[str] = set()
    parts: list[str] = []
    for p in passages[:_CONTEXT_K]:
        text = (p.get("text") if isinstance(p, dict) else str(p)) or ""
        if text and text not in seen:  # drop exact-duplicate passages
            seen.add(text)
            parts.append(text)
    ctx = "\n\n---\n\n".join(parts)
    return ctx[:_CONTEXT_CHAR_BUDGET]


async def _regrade(input_path: Path, output_path: Path, *, limit: int | None) -> None:
    from dotenv import load_dotenv

    from evaluation.scoring.dspy_metrics import DspyMetricScorer

    load_dotenv(override=True)
    data = json.loads(input_path.read_text())
    cases = data.get("cases", [])
    if limit:
        cases = cases[:limit]

    scorer = DspyMetricScorer()  # OpenRouter LM from env
    sf1_all, cg_all, comp_all, grnd_all = [], [], [], []
    rows = []
    for i, case in enumerate(cases, 1):
        q = case.get("question", "")
        gold = case.get("gold", "")
        pred = case.get("generated_answer", "")
        ctx = _context_from_passages(case)
        scores = await scorer.score(q, gold, pred, ctx)
        case.setdefault("metrics", {}).update(scores.as_metrics())
        sf1_all.append(scores.semantic_f1)
        cg_all.append(scores.complete_and_grounded)
        comp_all.append(scores.completeness)
        grnd_all.append(scores.groundedness)
        rows.append(
            {
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "gold": gold,
                "generated_answer": pred,
                "token_f1": round(case.get("score", 0.0), 4),
                "semantic_f1": round(scores.semantic_f1, 4),
                "complete_and_grounded": round(scores.complete_and_grounded, 4),
                "completeness": round(scores.completeness, 4),
                "groundedness": round(scores.groundedness, 4),
            }
        )
        print(
            f"  [{i:3d}/{len(cases)}] {case.get('case_id')}  "
            f"semF1={scores.semantic_f1:.3f}  C&G={scores.complete_and_grounded:.3f}  "
            f"(compl={scores.completeness:.2f} grnd={scores.groundedness:.2f})",
            flush=True,
        )

    data["dspy_metric_summary"] = {
        "n": len(cases),
        "mean_semantic_f1": round(st.mean(sf1_all), 4) if sf1_all else 0.0,
        "mean_complete_and_grounded": round(st.mean(cg_all), 4) if cg_all else 0.0,
        "mean_completeness": round(st.mean(comp_all), 4) if comp_all else 0.0,
        "mean_groundedness": round(st.mean(grnd_all), 4) if grnd_all else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{json.dumps(data['dspy_metric_summary'], indent=2)}")
    print(output_path)
    print(csv_path)


def _selftest() -> int:
    """Exercise the metric wiring offline with a stub LM (no network)."""
    import dspy
    from dspy.utils.dummies import DummyLM

    from evaluation.scoring.dspy_metrics import DspyMetricScorer

    # Canned LM outputs, consumed in call order: SemanticF1, then completeness,
    # then groundedness. Values chosen so the arithmetic is checkable by hand.
    lm = DummyLM(
        [
            {"reasoning": "r", "recall": 0.5, "precision": 1.0},  # SemanticF1 -> f1(1.0,0.5)=0.667
            {
                "reasoning": "r",
                "ground_truth_key_ideas": "a",
                "system_response_key_ideas": "b",
                "discussion": "d",
                "completeness": 0.5,
            },
            {
                "reasoning": "r",
                "system_response_claims": "c",
                "discussion": "d",
                "groundedness": 0.8,
            },  # C&G -> f1(0.8,0.5)=0.615
        ]
    )
    scorer = DspyMetricScorer(lm=lm)
    scores = asyncio.run(
        scorer.score(
            "When did Caroline go to the LGBTQ support group?",
            "7 May 2023",
            "Unknown",
            "some retrieved passage text",
        )
    )
    ok = (
        abs(scores.semantic_f1 - 0.6667) < 1e-3
        and abs(scores.complete_and_grounded - 0.6154) < 1e-3
    )
    print("selftest scores:", scores.as_metrics())
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="results JSON to re-grade")
    p.add_argument("--output", type=Path, help="output JSON path (CSV written alongside)")
    p.add_argument("--limit", type=int, default=None, help="grade only the first N cases")
    p.add_argument("--selftest", action="store_true", help="offline wiring test, no network")
    args = p.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.input or not args.output:
        p.error("--input and --output are required unless --selftest")
    asyncio.run(_regrade(args.input, args.output, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
