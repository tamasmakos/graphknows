"""LoCoMo prompts: staged-reasoning answer generation + binary judge (semantic leniency, strict dates).

Design notes (each choice is load-bearing):

* Memories are shown **chronologically with dates and no ranks/scores** —
  score anchoring biases the answerer toward the top hit, and temporal
  questions need the timeline visible.
* The answer prompt walks fixed reasoning stages (scan → attribute → combine →
  select → time-ground → include → commit) because the dominant failure modes
  on this benchmark are early-stopping on the first relevant memory, dropped
  list items, and relative-date mistakes.
* The judge is **binary, semantically lenient, temporally strict**: paraphrase,
  partial list credit, and same-referent matches still count, but dates carry NO
  tolerance — after resolving relative references, the answer must denote the
  exact gold day. (The former ±14-day window both masked real temporal misses
  and was applied inconsistently by the judge model.) Lexical/token F1 punishes
  verbose-but-correct answers; the binary judge measures whether the memory
  system recalled the right fact, which is the quantity under test.
"""

ANSWER_SYSTEM = ""  # single user-message prompt; keeps provider routing simple

ANSWER_PROMPT = """\
You are answering a question about long-running conversations between two people, using \
retrieved memories from those conversations. Work through the stages below IN ORDER.

## Stage 1 — SCAN EVERYTHING
Read every memory from first to last and note each one relevant to the question. Do NOT \
stop at the first relevant memory: supporting details are often scattered, and a memory \
near the end is exactly as likely to hold the answer as one near the start.
Match MEANING, not wording: the memories almost never reuse the question's exact words. \
"Signed up to be a mentor for LGBTQ youth" answers a question about "joining a mentorship \
program"; "sent in applications to agencies" answers one about "applying". A memory is \
relevant when it describes the same real-world event in any phrasing.

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

**THE MESSAGE DATE IS NOT THE EVENT DATE.** Every memory is prefixed with the date it was \
SPOKEN. A past-tense report describes something that already happened, so the event is \
EARLIER than the message date. Answering with the message date is the single most common \
error on these questions — do not do it unless the text says the event happened that same \
day ("today", "this morning", "just now").

Work each date question in three explicit steps:
1. Find the message date D that carries the report (its "(Month DD, YYYY)" prefix).
2. Read the time expression in the sentence itself — "yesterday", "last weekend", "last \
week", "a couple of weeks ago", "last month", "last year", "on Tuesday".
3. SUBTRACT it from D and answer with the result, not with D.

   "(July 17, 2023) … I joined a mentorship program last weekend" → the weekend BEFORE \
July 17 → answer 15 July 2023. NOT 17 July.
   "(May 25, 2023) … I went camping yesterday" → 24 May 2023. NOT 25 May.
   "(June 9, 2023) … I met up with them last week" → the week before 9 June 2023.

- If the sentence carries NO time expression and is past tense, the event is shortly \
before D — say so relative to D rather than asserting D itself.
- Future tense ("I'm going camping next month") → compute FORWARD from D: from (May 25, \
2023), "next month" is June 2023, not May.
- Compute relative time against the conversation dates, never against today.
- Use dates stated in the memories; never invent one.
- If several similar events exist at different dates, enumerate them with dates before \
choosing: past tense + "the" → the instance closest before the reference date; future \
tense ("plans to") → the earliest planned date.
- Give dates at day/month/year granularity — never carry a clock time into the answer. \
When the memories pin down the DAY (stated outright, or derivable from a message date \
plus "yesterday"/"last weekend"), answer with the full day-month-year; "July 2023" when \
"10 July 2023" is derivable loses credit for imprecision.
- Only say a date is absent after checking EVERY memory for a message-date + time \
expression pair. A report with a date prefix always grounds SOME date.

## Stage 6 — INCLUSION CHECK (lists and counts)
If you found an item during reasoning, include it unless you have strong evidence it is \
wrong — the most common error is finding items and then over-filtering them. After \
enumerating, deduplicate (one event described two ways) and re-check the tail of the \
memory list for missed items.

## Stage 7 — COMMIT
Answer directly and specifically. Never reply "not specified" or "the memories don't \
say" when ANY memory is relevant — give the best supported answer. Before concluding \
that information is absent, re-run Stage 1 once looking ONLY for paraphrases of the \
question's event: absence conclusions are the most common error and are almost always a \
missed rewording, not missing information. Never output a name, \
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
4. DATES ARE STRICT — NO TOLERANCE: first resolve any relative reference to an absolute \
date (gold "the Sunday before 25 May 2023" means 21 May 2023; resolving "last year" to \
the actual year is required, not penalized). After resolution, the generated answer must \
denote the SAME day: 20 May 2023 against a gold of 21 May 2023 is WRONG. If the gold \
gives only a month or year, matching at that granularity suffices; if the gold specifies \
a day, the generated answer must give that exact day. Durations must match exactly \
("5 months" vs "six months" is WRONG).
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
