"""LongMemEval prompts: assistant-memory answerer + lenient yes/no judge.

The answer prompt anchors all relative time to the question date, forces an
explicit reasoning pass inside ``<mem_thinking>`` tags (counting and temporal
questions fail without enumerate-then-answer discipline), and defines a single
canonical abstention phrase so abstention questions are judgeable. The judge is
binary yes/no with an explicit anti-"no" bias check — graders systematically
under-credit differently-worded correct answers.
"""

ANSWER_SYSTEM = ""

ANSWER_PROMPT = """\
You are a personal assistant with memories from past conversations with a user. Answer \
the question from the memories below. Be direct and concise.

Today's date is {question_date}. Compute ALL relative time expressions ("last month", \
"how many days ago") against this date, never against the real today.

Before answering, reason inside <mem_thinking> tags:
- List every memory relevant to the question — scan the full list; relevant items are \
often far down.
- For counting questions: enumerate each distinct instance with its date, apply the \
question's exact verb and qualifier strictly (completed actions only, not plans), \
deduplicate same-event descriptions, THEN count the list. Re-scan once for missed items.
- For temporal questions: write out the dates and the interval arithmetic. Time-bounded \
windows are inclusive; compute the window first, then test every item's date against it.
- Check each memory's context matches the question's context (a routine "while \
traveling" is not the weekday routine); prefer the more specific matching memory.
- For cross-topic computations (ages, prices, durations): the needed facts are often in \
unrelated conversations — hunt for each fact independently, then do the arithmetic. \
Never abstain when the raw numbers exist scattered across memories.

Rules:
1. Most recent wins: when memories give conflicting values for the same fact, use the \
latest one — a current count/status REPLACES earlier values (don't sum them). Direct \
contradictions: trust the later memory.
2. Facts stated by the user are ground truth about the user; things the assistant \
suggested or explained are advice, not user facts. For personal questions use \
user-stated facts.
3. Actions vs intentions: use the date something was actually done, not when it was \
planned. Explicit future tense ("plans to", "will") is a plan; a plan with a stated \
date and no later update can be assumed executed on that date.
4. Entity precision: if the question names a specific variant, role, or title and the \
memories only support a DIFFERENT one (e.g. "Sales Manager" asked, "Senior Sales \
Engineer" stored), do not treat them as the same — abstain.
5. For recommendation/suggestion questions, personalize: ground every suggestion in the \
user's stated preferences, tools, and habits; check each against their known dislikes; \
lead with what fits them rather than generic options. Don't abstain for lack of \
real-time information.
6. Use only the memories — never invent names, numbers, prices, or dates. If you have a \
description but no name, answer with the description.
7. Yes/no history questions: "Did I ever do X?" with no supporting memory = "No."
8. Abstain — answering exactly "The information provided is not enough" — only when the \
topic is genuinely absent, the asked-about event does not exist in memory, or rule 4 \
applies. Before abstaining, do one keyword re-scan of ALL memories.

Memories (oldest first, each prefixed with its date):
{memories}

Today's date: {question_date}
Question: {question}

Reason in <mem_thinking> first (the user sees only text outside the tags), then give \
your final answer on a new line after "ANSWER:".\
"""

JUDGE_PROMPT = """\
I will give you a question, a correct answer (or grading rubric), and a model response. \
Decide whether the model response is correct.

CORE PRINCIPLE — judge by MEANING, not wording. The response is correct when every \
concept in the correct answer is addressed, even with different vocabulary, more \
specific terms, or restructured phrasing.

BIAS CHECK: graders say "no" too quickly. Before concluding "no", verify the response \
is actually wrong rather than differently worded. When genuinely in doubt, lean "yes".

Rules:
- Supersets are correct: extra details, extra qualifiers, or added scope notes are fine \
unless they contradict the correct answer.
- Lists: match items by meaning; synonyms and sub-concepts cover an item. Optional \
("or"/"maybe") items may be omitted. Two listed items serving the same purpose — naming \
one is enough.
- Numbers: hedges ("about", "at least") and ranges containing the correct value are \
correct; more precise values matching the coarse answer are correct ("22 days" ≈ "3 \
weeks"); off-by-one on day/week/month arithmetic is acceptable; unit conversions may \
round ("14 weeks" ≈ "3 months").
- Dates: format variations are equivalent; a correct value with supplementary context in \
parentheses is correct; colloquial windows ("last weekend", "last month") are imprecise \
— allow boundary flexibility.
- Counting edge case: gold "0"/"nothing found" matches a response of "not enough \
information", and vice versa.
- Preference/personalization rubrics: correct when the response demonstrates awareness \
of the user's personal context and its main suggestions align with what the user wants; \
the rubric is a guide, not a checklist. Judge anti-preferences by overall thrust, not \
keyword scanning.
- Abstention matching applies ONLY when the CORRECT ANSWER is itself unanswerable/an \
abstention: in that case any phrasing that declines to answer ("I don't have that \
information", "no record of X") is correct. When the correct answer is a concrete fact, \
a response that declines to answer or says "not enough information" is WRONG — a \
retrieval failure is not a correct answer.

FINAL CHECK before answering "no": (1) what is the core claim of the correct answer? \
(2) does the response address that claim in any wording? (3) is it a superset? (4) do \
core numbers match modulo hedging? (5) for abstention gold, does the response decline? \
Only answer "no" if a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {gold}

Model Response: {answer}

Think step-by-step inside <judge_thinking> tags, then give your final verdict as \
exactly "yes" or "no" on a new line after the closing tag.\
"""
