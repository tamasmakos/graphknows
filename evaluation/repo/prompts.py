"""Repository-transcript answer prompt.

The memories are turns and tool calls from an agent working in this repository,
so the answer is a fact about the WORK: where a decision was made, why a line
changed, which test guards a symbol. Judging uses the shared DSPy judge, so no
judge prompt lives here.
"""

ANSWER_SYSTEM = ""  # single user-message prompt; keeps provider routing simple

ANSWER_PROMPT = """\
You are answering a question about work done in a software repository, using retrieved \
memories from the agent transcripts of that work.

Rules:
- Ground every claim in the memories. Name the file, symbol, test or commit the memories \
name, spelled exactly as they spell it.
- A decision is where it was SETTLED, not everywhere it was discussed.
- A rationale is the reason actually given at the time, not one you can reconstruct.
- Sessions are ordered oldest first; the reference date is {reference_date}. Resolve \
relative time against the memory's own date.
- If the memories genuinely do not answer the question, say so plainly rather than \
inventing a path, symbol or reason.

Memories (oldest first):
{memories}

Question: {question}

Give your final answer on a new line after "ANSWER:".\
"""
