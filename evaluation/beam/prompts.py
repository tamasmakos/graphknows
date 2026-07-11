"""BEAM prompts: rule-based answerer + per-nugget rubric judge.

BEAM grades against rubric nuggets, so the judge runs once per nugget and
returns a 0.0/0.5/1.0 compliance score; the question score is the nugget mean.
Event-ordering questions additionally extract the answer's event sequence and
align it to the rubric order for a Kendall tau-b term.
"""

ANSWER_SYSTEM = ""

ANSWER_PROMPT = """\
You are an AI assistant with stored memories from prior conversations with a user.
Use those memories to answer the question as accurately and completely as possible.

Rules:
1. Scan ALL memories before answering — never stop at the first relevant one.
2. Combine and cross-reference every memory that bears on the question.
3. On contradictions, prefer the most recent memory.
4. If the memories cannot support an answer, say exactly: "I don't have enough \
information to answer this question."
5. Temporal questions: use the dates on the memories; compute intervals explicitly.
6. Ordering questions: present events in chronological order.
7. Preference questions: use the most recently stated preference.
8. Instruction-following: apply any standing instructions the user gave earlier.
9. Be specific — include exact names, dates, numbers, and details from the memories, \
and never invent information that is not in them.

QUESTION: {question}

MEMORIES (oldest first, each prefixed with its date):
{memories}

Give your final answer after "ANSWER:".\
"""

NUGGET_JUDGE_SYSTEM = (
    "You are an expert evaluator checking whether a response satisfies a rubric "
    "criterion. Be objective and consistent. Return ONLY valid JSON in the exact "
    "format requested."
)

NUGGET_JUDGE_PROMPT = """\
Evaluate whether the response complies with the rubric criterion.

QUESTION:
{question}

RESPONSE:
{answer}

RUBRIC CRITERION:
{nugget}

First classify the criterion as a POSITIVE requirement (the response should contain or
demonstrate something) or a NEGATIVE constraint (the response should avoid something).

POSITIVE requirement scoring:
- 1.0 — the required element is present, accurate, and complete.
- 0.5 — partially present, minor inaccuracies, or incomplete.
- 0.0 — missing, incorrect, or the response is off-topic/non-responsive.

NEGATIVE constraint scoring:
- 1.0 — the response answers the question AND the prohibited element is absent.
- 0.5 — responsive, but a borderline/ambiguous reference to the prohibited element.
- 0.0 — the prohibited element is present, OR the response is non-responsive.

Compound criteria (elements joined by "and" or commas): all elements satisfied = 1.0,
some = 0.5, none = 0.0.

Evaluation rules:
1. Semantic tolerance — paraphrases and synonyms satisfy the criterion.
2. Numeric/date equivalence — "$68,000" = "68k"; "2 years" = "24 months"; compare
   normalized values.
3. Ignore case, punctuation, and whitespace differences.
4. Do not penalize hedging, passive voice, or verbosity when the substance complies.
5. Ignore tone/format/length unless the criterion requires a specific format.
6. A refusal or fully off-topic response scores 0.0.
7. Judge this criterion in isolation — ignore the other rubric items.
8. Specific, detailed answers outrank vague ones that could fit any question.

Steps: understand the criterion → split compound elements → check compliance with the
tolerance rules → assign the score → explain in one sentence.

Return JSON with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence>"}}\
"""

EVENT_EXTRACTION_SYSTEM = "Extract events as a JSON array of strings."

EVENT_EXTRACTION_PROMPT = """\
Extract all distinct events or facts mentioned in the following response, in the exact
order they are presented. Return ONLY a JSON array of short event descriptions.

RESPONSE:
{answer}

Return format: ["event 1 description", "event 2 description", ...]\
"""

EVENT_ALIGNMENT_SYSTEM = "Align the event to a reference event index. Return JSON."

EVENT_ALIGNMENT_PROMPT = """\
Given the following event extracted from a response, determine which reference event it
best corresponds to. Return ONLY a JSON object.

EXTRACTED EVENT:
{event}

REFERENCE EVENTS:
{events_list}

If the extracted event matches one of the reference events (even approximately or
paraphrased), return its 0-based index. If it matches none, return -1.

Return format: {{"index": <integer>, "reason": "<brief explanation>"}}\
"""
