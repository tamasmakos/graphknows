"""Compare two locomo eval run JSON files and produce a side-by-side results record.

Usage:
    python evaluation/scripts/compare_runs.py \
        --run-a evaluation/results/locomo/smoke10-gliner2-locomo.json \
        --run-b evaluation/results/locomo/smoke10-llm_assisted-locomo.json \
        --label-a gliner2 \
        --label-b llm_assisted \
        --output research/locomo_smoke10_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["case_id"]: c for c in data.get("cases", [])}


def _top_recall(case: dict, n: int = 2) -> list[str]:
    hits = case.get("memory_context", [])
    return [h.get("text", "")[:150].replace("\n", " ") + "..." for h in hits[:n]]


def compare(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    output: Path,
) -> None:
    cases_a = load_cases(path_a) if path_a.exists() else {}
    cases_b = load_cases(path_b) if path_b.exists() else {}

    all_ids = list(cases_a.keys() or cases_b.keys())
    rows = []
    for case_id in all_ids:
        ca = cases_a.get(case_id, {})
        cb = cases_b.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "dialogue_id": ca.get("metadata", {}).get("dialogue_id")
                or cb.get("metadata", {}).get("dialogue_id"),
                "category": ca.get("metadata", {}).get("category")
                or cb.get("metadata", {}).get("category"),
                "question": ca.get("question") or cb.get("question"),
                "gold": ca.get("gold") or cb.get("gold"),
                label_a: {
                    "score": ca.get("score"),
                    "score_type": ca.get("metrics", {}).get("score_type"),
                    "generated_answer": ca.get("metadata", {}).get("generated_answer"),
                    "recall_hits": len(ca.get("memory_context", [])),
                    "ltm_graph_hits": ca.get("metrics", {}).get("ltm_graph_hit_count", 0),
                    "top_recall_snippets": _top_recall(ca),
                },
                label_b: {
                    "score": cb.get("score"),
                    "score_type": cb.get("metrics", {}).get("score_type"),
                    "generated_answer": cb.get("metadata", {}).get("generated_answer"),
                    "recall_hits": len(cb.get("memory_context", [])),
                    "ltm_graph_hits": cb.get("metrics", {}).get("ltm_graph_hit_count", 0),
                    "top_recall_snippets": _top_recall(cb),
                },
            }
        )

    scores_a = [r[label_a]["score"] for r in rows if r[label_a]["score"] is not None]
    scores_b = [r[label_b]["score"] for r in rows if r[label_b]["score"] is not None]

    result = {
        "run_a": {"label": label_a, "source": str(path_a)},
        "run_b": {"label": label_b, "source": str(path_b)},
        "summary": {
            label_a: {
                "avg_score": round(sum(scores_a) / len(scores_a), 4) if scores_a else None,
                "scored_cases": len(scores_a),
            },
            label_b: {
                "avg_score": round(sum(scores_b) / len(scores_b), 4) if scores_b else None,
                "scored_cases": len(scores_b),
            },
            "total_cases": len(rows),
            "unscored_categories": ["open_ended"],
        },
        "cases": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} cases to {output}")
    print(
        f"\n{label_a}: avg={result['summary'][label_a]['avg_score']} (n={result['summary'][label_a]['scored_cases']})"
    )
    print(
        f"{label_b}: avg={result['summary'][label_b]['avg_score']} (n={result['summary'][label_b]['scored_cases']})"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--label-a", default="run_a")
    p.add_argument("--label-b", default="run_b")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    compare(Path(args.run_a), Path(args.run_b), args.label_a, args.label_b, Path(args.output))


if __name__ == "__main__":
    main()
