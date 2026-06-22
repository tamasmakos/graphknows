"""DSPy signature + few-shot demos for LoCoMo answer generation.

The official LoCoMo paper (arXiv 2402.17753) scores token F1 between a
model-generated answer and the gold answer. This signature drives that
generation step so our scores are paper-comparable.
"""

from __future__ import annotations

import dspy

from evaluation.datamodels import RetrievedPassage

_MAX_CONTEXT_CHARS = 8_000


class LoCoMoAnswerGeneration(dspy.Signature):
    """Answer a question about a long-running conversation between two people,
    using ONLY the retrieved conversation passages as the source of truth.

    Rules:
    1. EXTRACT, do not abstain. If any passage supports an answer — even stated
       indirectly or via light inference — give that answer. Say "Unknown" ONLY
       when no passage is topically related to the question.
    2. Shortest correct answer: a name, date, place, number, or comma-separated
       list. No framing words ("Sweden" not "my home country, Sweden"). For yes/no
       inference questions, add a brief reason ("Yes, since she plays piano").
    3. Resolve dates: convert purely relative phrases (e.g. "last year" → "2022",
       "next month" → "June 2023"). But preserve period-anchored expressions as-is
       (e.g. "the week before 27 June 2023" — do NOT reduce to "20 June 2023").
       Resolve vague entity references ("my home country" → "Sweden").
    4. Multi-item questions — list EVERY matching item found across all passages,
       comma-separated and deduplicated."""

    question: str = dspy.InputField(desc="A question about the two-person conversation.")
    retrieved_context: str = dspy.InputField(
        desc="Numbered conversation passages, ordered by relevance. The only source of truth."
    )
    answer: str = dspy.OutputField(
        desc=(
            "Shortest correct answer: a name, date, place, number, or comma-separated list "
            "(all references resolved); or a qualified yes/no for inference questions "
            "('Yes, since she plays piano' / 'Likely no; since it went poorly'); "
            "or 'Unknown' only if no passage is topically related."
        )
    )


def demos() -> list[dspy.Example]:
    """Few-shot demos teaching extraction, inference, list aggregation, and date anchoring.

    These use invented people/facts that do NOT appear in the benchmark — they
    teach answer *behaviour and format*, never leak benchmark answers.
    """
    raw = [
        # Direct extraction — answer stated indirectly, do not over-abstain.
        (
            "What is Daniel's profession?",
            "[1] Daniel: After residency I finally settled in as a paediatric nurse "
            "at the city hospital. Daniel: the night shifts are tough but I love it.",
            "Paediatric nurse",
        ),
        # Strip framing words — answer the bare value only.
        (
            "Which country is Sven from?",
            "[1] Sven: honestly my home country, Norway, is where I feel most at peace.",
            "Norway",
        ),
        # Multi-item aggregation across passages.
        (
            "What instruments does Tomás play?",
            "[1] Tomás: I practised the cello this morning.\n\n[2] Tomás: jammed on "
            "guitar with friends.\n\n[3] Tomás: piano is still my first love.",
            "cello, guitar, piano",
        ),
        # Relative date — resolve to absolute from the timestamp in the passage.
        (
            "When did Priya adopt her dog?",
            "[1] 12 March 2023 | Priya: I adopted Bella yesterday, she's settled "
            "in so well already.",
            "11 March 2023",
        ),
        # Positive inference — infer from behavioral evidence and include brief reason.
        (
            "Would Alex consider himself athletic?",
            "[1] Alex: Ran my first 5K last Saturday, finished in 28 minutes!\n\n"
            "[2] Alex: Gym three times a week now — really feeling the progress.",
            "Yes, since he runs regularly and goes to the gym",
        ),
        # Negative inference — do NOT abstain; infer from failure/obstacle context.
        (
            "Would Marcus want to return to the same resort?",
            "[1] Marcus: That resort was honestly a disaster — dirty rooms, terrible food. "
            "I'd never book there again.",
            "Likely no; since the experience was very negative",
        ),
    ]
    return [
        dspy.Example(question=q, retrieved_context=c, answer=a).with_inputs(
            "question", "retrieved_context"
        )
        for q, c, a in raw
    ]


def build_context(passages: list[RetrievedPassage]) -> str:
    """Build a numbered, deduped passage block clamped to ``_MAX_CONTEXT_CHARS``."""
    lines: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        text = passage.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(f"[{len(lines) + 1}] {text}")
    return "\n\n".join(lines)[:_MAX_CONTEXT_CHARS]
