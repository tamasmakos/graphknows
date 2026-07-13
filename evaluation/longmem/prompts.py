"""LongMemEval answer prompt.

The haystack is a user/assistant chat history spread across many sessions; the
answer must be grounded in what the *user* actually said (or the assistant
established) and time-anchored to the session dates. Judging uses the shared
DSPy judge, so no judge prompt lives here.
"""

ANSWER_SYSTEM = ""  # single user-message prompt; keeps provider routing simple

ANSWER_PROMPT = """\
You are answering a question about a user, using retrieved memories from that user's \
past chat sessions with an assistant. Work through the stages below IN ORDER.

## Stage 1 — SCAN EVERYTHING
Read every memory from first to last. Evidence is often in a single earlier session; a \
memory near the end is as likely to hold the answer as one near the start. Do not stop at \
the first relevant memory.

## Stage 2 — CHECK ATTRIBUTION
Keep straight who said what. A preference or fact the USER stated is about the user; \
something the assistant merely suggested is not the user's own position unless the user \
adopted it.

## Stage 3 — COMBINE ACROSS SESSIONS
- Merge facts about the same thing stated across different sessions.
- KNOWLEDGE UPDATES: when the user revises a fact ("I moved to Berlin"), the most RECENT \
statement wins — report the current state, not the outdated one.
- For list/counting questions, gather EVERY distinct item across ALL memories before \
counting.

## Stage 4 — GROUND EVERY DATE
Each memory is prefixed with its session date. Compute relative time ("last month", "two \
weeks ago") against the SESSION date it was said in, never against today. Give dates at \
day/month/year granularity.

## Stage 5 — COMMIT
Answer directly and specifically. If the memories genuinely contain no information to \
answer the question, say so plainly rather than inventing a fact — but do not refuse when \
any memory is relevant. Never output a name, date, or number that appears in no memory.

Memories (oldest first, each prefixed with its session date):
{memories}

Question: {question}

Work through Stages 1-5, then give your final answer on a new line after "ANSWER:".\
"""
