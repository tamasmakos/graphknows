# Repository-transcript gold set (FR-044)

`gold.jsonl` is the hand-written gold set for the repository row, loaded by
[`evaluation/repo/dataset.py`](../../repo/dataset.py). One JSON object per line:

```json
{"case_id": "repo-t065-tests", "transcript": "transcripts/session-fixture.jsonl",
 "kind": "test_coverage", "commit": "8f8229a",
 "question": "Which test guards that a pack's labels replace the core inventory rather than extend it?",
 "answer": "tests/extraction/test_pack_guidance.py, added by commit 8f8229a.",
 "evidence": ["tests/extraction/test_pack_guidance.py is the test that guards it - it asserts a pack's labels replace rather than extend."]}
```

`evidence` holds the transcript sentences the answer is read off, verbatim, so
evidence recall is scored against the same text that was ingested. `commit` is
the commit in this repository's history the answer was checked against: each
answer is derived from the transcript and from `git log`, never from one alone.

## Kinds

Three, one question each per transcript:

- `decision_location` — where a thing was settled, not everywhere it was discussed;
- `change_rationale` — the reason actually given at the time;
- `test_coverage` — which test guards a symbol.

## Corpus

The row's corpus is the user's own agent transcripts, read at run time from the
Claude Code harness directory for this checkout (`~/.claude/projects/<slugged
working directory>`, see `claude_code_sessions_root` in
[`evaluation/repo/config.py`](../../repo/config.py)); session content is not
committed.

`transcripts/` holds one fixture, `session-fixture.jsonl`, and it is **written for
this test, not a recorded session** — committing session content would put a
developer's working conversation into the repository's history. What it copies
from the wild is the *shape*: the same record types in the same order, harness
bookkeeping (`mode`, `permission-mode`, `file-history-snapshot`, `system`) beside
the turns, and the full sixteen-field turn record (`uuid`, `parentUuid`,
`sessionId`, `cwd`, `gitBranch`, `isSidechain`, `version`, …) that a transcript
written as plain prose would not have. So the loader is exercised on the shape it
meets in the wild while CI stays deterministic, and the row's real measurement
still comes from the run-time session directory above. A gold record names a
transcript; it is read from here when it is the committed fixture and from the
session directory otherwise.

## Invariants

`tests/evaluation/test_repo_row_kinds.py` checks the gold set: three kinds
covered, every case's transcript present, every evidence sentence found verbatim
in the document its case is answered from, every path an answer names present in
the tree. `tests/evaluation/test_repo_row_is_real.py` checks the corpus: one
committed fixture carrying the real record shape and its bookkeeping, every
answer's commit known to `git`, and an uncommitted transcript read from the
session directory.

Adding a case is a hand edit of `gold.jsonl`.
