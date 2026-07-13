"""BEAM answer prompt.

BEAM probes ten task types over a large conversation history. The answer must be
grounded in retrieved memory and handle the harder modes the benchmark targets:
abstain when the memory genuinely lacks the answer, honor stated instructions and
preferences, resolve contradictions by recency, and reason across sessions.
Grading uses the shared DSPy judges, so no judge prompt lives here.
"""

ANSWER_SYSTEM = ""  # single user-message prompt; keeps provider routing simple

ANSWER_PROMPT = """\
You are answering a question about a user's long conversation history with an assistant, \
using retrieved memories from that history. Work through the stages below IN ORDER.

## Stage 1 — SCAN EVERYTHING
Read every memory. Evidence for one answer is often spread across sessions; a memory near \
the end is as likely to matter as one near the start.

## Stage 2 — COMBINE ACROSS SESSIONS
- Merge and cross-reference facts stated in different sessions.
- KNOWLEDGE UPDATES / CONTRADICTIONS: when the history states conflicting facts, the most \
RECENT statement is the current truth — report it, and note the change if asked.
- For ordering questions, place events on a timeline before answering.

## Stage 3 — GROUND EVERY DATE
Each memory is prefixed with its time anchor. Compute relative time against the anchor of \
the session it was said in, never against today.

## Stage 4 — FOLLOW INSTRUCTIONS & PREFERENCES
If the user previously set an instruction or preference about how to answer (format, \
tone, constraints), honor it in your answer.

## Stage 5 — ABSTAIN WHEN UNSUPPORTED
If the retrieved memories genuinely do not contain the information needed, say so plainly \
("I don't have that information") rather than guessing. Do NOT fabricate names, dates, or \
numbers that appear in no memory. But do not refuse when the memories do answer it.

Memories (oldest first, each prefixed with its time anchor):
{memories}

Question: {question}

Work through Stages 1-5, then give your final answer on a new line after "ANSWER:".\
"""
