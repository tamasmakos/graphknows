"""Convert the official LoCoMo HuggingFace dataset to graphknows evaluation format.

Source: KimmoZZZ/locomo on HuggingFace (10 dialogues, 1,986 QA pairs)
Official structure:
  - One row per dialogue with fields: sample_id, qa, conversation, event_summary,
    observation, session_summary
  - conversation: dict with speaker_a/b and session_1..N (list of turns) +
    session_N_date_time keys
  - Turn format: {"speaker": "Name", "dia_id": "D1:3", "text": "..."}
    (some turns also have img_url, blip_caption, query for images)
  - qa: list of {question, answer, evidence, category}
    category codes: 1=single_hop, 2=temporal, 3=open_ended, 4=multi_hop, 5=adversarial
    Category 5 uses "adversarial_answer" instead of "answer" (no correct answer)

Output: graphknows JSONL — one row per QA pair:
  {id, category, question, answer, sessions: [{session_id, messages: [{role, content, timestamp}]}]}

Usage:
    python evaluation/scripts/convert_locomo.py \
        --output evaluation/data/locomo/questions.jsonl \
        [--include-adversarial]  # cat 5 defaults to excluded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Category mapping derived from evidence pattern analysis and paper description.
# Cat 1: questions synthesizing facts from many sessions (2-19+ evidence refs)
# Cat 2: temporal questions ("when did X happen")
# Cat 3: open-ended / commonsense inference (answer is not stated verbatim)
# Cat 4: single-event factual questions (94.5% have exactly 1 evidence ref)
# Cat 5: adversarial (no correct answer; adversarial_answer field only)
CATEGORY_NAMES = {
    1: "knowledge_synthesis",
    2: "temporal",
    3: "open_ended",
    4: "single_hop",
    5: "adversarial",
}

MAX_SESSIONS = 50  # safety cap matching --locomo-max-documents default


def _build_sessions(conv: dict, sample_id: str) -> list[dict]:
    """Convert conversation dict into graphknows sessions list."""
    sessions = []
    for i in range(1, MAX_SESSIONS + 1):
        session_key = f"session_{i}"
        turns = conv.get(session_key)
        if not isinstance(turns, list):
            break
        date_time = conv.get(f"session_{i}_date_time", "")
        messages = []
        for turn in turns:
            text = turn.get("text", "").strip()
            # Image turns: supplement missing text with BLIP caption + search query.
            if not text:
                caption = turn.get("blip_caption", "")
                query = turn.get("query", "")
                parts = []
                if caption:
                    parts.append(f"[image: {caption}]")
                if query:
                    parts.append(f"[search: {query}]")
                text = " ".join(parts)
            if not text:
                continue
            msg: dict = {
                "role": turn.get("speaker", ""),
                "content": text,
            }
            if date_time:
                msg["timestamp"] = date_time
            messages.append(msg)
        if messages:
            sessions.append(
                {
                    "session_id": f"{sample_id}-s{i}",
                    "messages": messages,
                }
            )
    return sessions


def convert(dataset_name: str, include_adversarial: bool) -> list[dict]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(dataset_name)
    rows: list[dict] = []

    for dialogue in ds["train"]:
        sample_id: str = dialogue["sample_id"]
        conv: dict = dialogue["conversation"]
        sessions = _build_sessions(conv, sample_id)

        for q_idx, qa in enumerate(dialogue["qa"], start=1):
            cat_num: int = qa.get("category", 0)

            if cat_num == 5 and not include_adversarial:
                continue

            category_name = CATEGORY_NAMES.get(cat_num, f"category_{cat_num}")

            if cat_num == 5:
                # Adversarial questions have no "answer"; system should abstain.
                answer = str(qa.get("adversarial_answer", "") or "")
            else:
                # Some answers are integers (year counts, numeric facts) — coerce to string.
                answer = str(qa.get("answer", "") or "")

            row = {
                "id": f"{sample_id}-q{q_idx:04d}",
                "category": category_name,
                "question": qa["question"],
                "answer": answer,
                "evidence": qa.get("evidence", []),
                "sessions": sessions,
            }
            rows.append(row)

    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="KimmoZZZ/locomo",
        help="HuggingFace dataset identifier (default: KimmoZZZ/locomo)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/data/locomo/questions.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include category 5 (adversarial) questions (excluded by default)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ...", flush=True)
    rows = convert(args.dataset, include_adversarial=args.include_adversarial)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    total = len(rows)
    cat_counts: dict[str, int] = {}
    for row in rows:
        cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1

    print(f"Wrote {total} rows to {out_path}")
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({100 * count / total:.1f}%)")


if __name__ == "__main__":
    main()
