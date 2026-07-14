"""LoCoMo prompts: staged-reasoning answer generation + lenient binary judge.

Design notes (each choice is load-bearing):

* Memories are shown **chronologically with dates and no ranks/scores** —
  score anchoring biases the answerer toward the top hit, and temporal
  questions need the timeline visible.
* The answer prompt walks fixed reasoning stages (scan → attribute → combine →
  select → time-ground → include → commit) because the dominant failure modes
  on this benchmark are early-stopping on the first relevant memory, dropped
  list items, and relative-date mistakes.
* The judge is **binary and lenient** (partial credit, paraphrase, superset,
  date tolerance). Lexical/token F1 punishes verbose-but-correct answers; the
  binary judge measures whether the memory system recalled the right fact,
  which is the quantity under test.
"""

ANSWER_SYSTEM = ""  # single user-message prompt; keeps provider routing simple

ANSWER_PROMPT = """\
You are answering a question about long-running conversations between two people, using \
retrieved memories from those conversations. Work through the stages below IN ORDER.

## Stage 1 — SCAN EVERYTHING
Read every memory from first to last and note each one relevant to the question. Do NOT \
stop at the first relevant memory: supporting details are often scattered, and a memory \
near the end is exactly as likely to hold the answer as one near the start.

## Stage 2 — CHECK ATTRIBUTION
Confirm each relevant memory is about the right person. If the question asks about person \
A and a memory states the fact about person B, do not transfer it to A. Both speakers' \
statements are valid evidence from their shared conversations — just keep the attribution \
straight.

## Stage 3 — COMBINE ACROSS MEMORIES
- Merge facts about the same event ("won first place" + "performed a piece titled X" \
describe one event).
- For list/counting questions, extract EVERY distinct item across ALL memories; a single \
memory can contribute several items. Enumerate each instance with its date or context \
BEFORE counting — list first, then count the list; never estimate.
- Decompose compound sentences: one sentence can contain several distinct candidate facts.
- Connect linked facts ("nearby lake" + "Lake Tahoe is great for kayaking" → the lake is \
Lake Tahoe; "bought it in Paris" → the country is France).

## Stage 4 — PICK THE BEST-SUPPORTED, MOST SPECIFIC ANSWER
- Retrieval order is NOT evidence. A lower-placed memory that answers the exact question \
beats a higher-placed one that is only near the topic.
- Prefer the most specific detail available: a proper name, title, or number beats a \
generic description. Repetition of a generic fact does not outrank one specific memory.
- Report what someone actually DID, not what was merely offered, planned, or "not tried \
yet".
- Photo captions describe the photo, not the person's life — prefer direct statements.
- Re-read the question: if it asks what "aspect"/"type"/"kind", answer with that specific \
aspect; answer the exact subject asked, not the broader family or household.

## Stage 5 — GROUND EVERY DATE
These conversations took place around {reference_date}; all events fall in 2022-2024.
- Compute relative time against the conversation dates, never against today.
- Use dates stated in the memories; never invent one.
- "Shared/mentioned X on date D" means they TALKED about it on D — the event is usually \
shortly before D.
- If several similar events exist at different dates, enumerate them with dates before \
choosing: past tense + "the" → the instance closest before the reference date; future \
tense ("plans to") → the earliest planned date.
- Give dates at day/month/year granularity — never carry a clock time into the answer.

## Stage 6 — INCLUSION CHECK (lists and counts)
If you found an item during reasoning, include it unless you have strong evidence it is \
wrong — the most common error is finding items and then over-filtering them. After \
enumerating, deduplicate (one event described two ways) and re-check the tail of the \
memory list for missed items.

## Stage 7 — COMMIT
Answer directly and specifically. Never reply "not specified" or "the memories don't \
say" when ANY memory is relevant — give the best supported answer. Never output a name, \
date, or number that appears in no memory. Reasonable deduction is allowed (a game \
exclusive to one platform implies owning that platform); invention is not.
- For hypothetical/opinion questions ("Would X do Y?"), follow the direct causal chain in \
the memories: doing Y because of Z means without Z, "likely no"; a recent bad experience \
outweighs an older positive pattern; partial evidence for a trait earns a qualified \
"somewhat", not a flat "no".

{facts}Memories (oldest first, each prefixed with its date):
{memories}

Question: {question}

Work through Stages 1-7, then give your final answer on a new line after "ANSWER:".\
"""

JUDGE_SYSTEM = (
    "You are grading whether a conversational memory system recalled the right fact. "
    "Return JSON only, in the exact format requested."
)

JUDGE_PROMPT = """\
Label the generated answer CORRECT or WRONG against the gold answer.

Grading rules — the question under test is whether the right fact was recalled, not \
whether the wording matches:

1. PARTIAL CREDIT: if the generated answer contains AT LEAST ONE item from the gold \
answer's list, label CORRECT (1 of 2, 2 of 4, … all count). Label WRONG only when NONE \
of the gold items appear.
2. PARAPHRASE COUNTS: the same concept in different words is CORRECT. Emotions in the \
same positive/negative family are paraphrases of each other ("proud" ≈ "fulfilled" ≈ \
"accomplished"). Judge meaning, not wording.
3. EXTRA DETAIL IS FINE: an answer containing the gold's key fact plus additional \
supported detail is CORRECT — never penalize for being more specific or more complete.
4. DATE TOLERANCE: dates within 14 days of each other match; durations within 50% match \
("5 months" ≈ "six months"). A specific date consistent with a vague gold reference \
matches, and resolving "last year" to the actual year is CORRECT.
5. SAME REFERENT: if both answers point at the same entity, person, or event — even \
described differently — label CORRECT.
6. SEMANTIC OVERLAP: different level of detail or scope is not WRONG if the core idea of \
the gold answer is captured.

Label WRONG only when the generated answer contains zero gold items or addresses a \
genuinely different topic.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT \
include both labels.\
"""
