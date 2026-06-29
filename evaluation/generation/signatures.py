"""DSPy signature + few-shot demos for LoCoMo answer generation.

The official LoCoMo paper (arXiv 2402.17753) scores token F1 between a
model-generated answer and the gold answer. This signature drives that
generation step so our scores are paper-comparable.
"""

from __future__ import annotations

import dspy

from evaluation.datamodels import RetrievedPassage

_MAX_CONTEXT_CHARS = 24_000


class LoCoMoAnswerGeneration(dspy.Signature):
    """Answer a question about a long-running conversation between two people,
    using ONLY the retrieved conversation passages as the source of truth.

    Reason fully over ALL the passages, then give the COMPLETE answer: include
    every fact, qualifier, and list item the passages support FOR THIS QUESTION,
    and nothing else. Match the gold's level of detail. Be concise — no framing,
    no explanation, no reasoning in the answer — but NEVER at the cost of dropping
    a required modifier or trailing clause. Both kinds of error count against you:
    a missing required item AND an extra unsupported/off-target item make the
    answer wrong.

    Rules:
    1. EXTRACT, do not abstain. If any passage supports an answer — even stated
       indirectly or via light inference — give it. Say "Unknown" ONLY when no
       passage is topically related at all.
    2. Strip framing words, keep substance. Give the bare value(s), but keep every
       qualifier that is part of the answer ("transgender woman", not "transgender";
       "counseling for transgender people", not "counseling"; "Sweden", not "my
       home country, Sweden"; "by dancing", not "They both like to destress by
       dancing together").
    3. Yes/no questions: answer "yes" or "no" alone — do NOT append a reason.
    4. Resolve dates: convert purely relative phrases ("last year" → "2022"). But
       preserve period-anchored expressions as-is ("the week before 27 June 2023" —
       do NOT reduce to "20 June 2023"). Resolve vague references ("my home country"
       → "Sweden").
    5. Multi-item questions — list EVERY matching item across all passages,
       comma-separated and deduplicated.
    6. ANSWER THE EXACT SUBJECT ASKED. If the question asks what someone's KIDS
       like, give only what the kids like — not what the whole family or the parent
       likes. Do not widen the scope to fill out a list.
    7. GROUND EVERY ITEM. Include an item ONLY if a passage explicitly states it for
       this subject. Never add plausible-but-unstated items (no "hiking" or
       "marshmallows" unless a passage says so). When unsure an item belongs, leave
       it out — a padded list scores worse than a short exact one."""

    question: str = dspy.InputField(desc="A question about the two-person conversation.")
    retrieved_context: str = dspy.InputField(
        desc="Numbered conversation passages, ordered by relevance. The only source of truth."
    )
    answer: str = dspy.OutputField(
        desc=(
            "The complete bare answer: every gold-relevant fact and qualifier with "
            "framing words stripped (a name, date, place, number, or comma-separated "
            "list, references resolved); 'yes'/'no' alone for yes/no questions; or "
            "'Unknown' only if no passage is topically related."
        )
    )


def demos() -> list[dspy.Example]:
    """Few-shot demos teaching extraction, inference, list aggregation, and date anchoring.

    These use invented people/facts that do NOT appear in the benchmark — they
    teach answer *behaviour and format* (concise, complete bare values), never leak answers.
    """
    raw = [
        # Direct extraction — answer stated indirectly, do not over-abstain.
        (
            "What is Daniel's profession?",
            "[1] Daniel: After residency I finally settled in as a paediatric nurse "
            "at the city hospital. Daniel: the night shifts are tough but I love it.",
            "paediatric nurse",
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
        # Yes/no inference — answer bare "yes", reason in thinking only, NOT in the answer.
        (
            "Would Alex consider himself athletic?",
            "[1] Alex: Ran my first 5K last Saturday, finished in 28 minutes!\n\n"
            "[2] Alex: Gym three times a week now — really feeling the progress.",
            "yes",
        ),
        # Negative inference — bare "no", do not abstain, no appended reason.
        (
            "Would Marcus want to return to the same resort?",
            "[1] Marcus: That resort was honestly a disaster — dirty rooms, terrible food. "
            "I'd never book there again.",
            "no",
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
