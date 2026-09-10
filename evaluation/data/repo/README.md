# Repository-transcript gold set (FR-044)

`gold.jsonl` is the hand-written gold set for the repository row, loaded by
[`evaluation/repo/dataset.py`](../../repo/dataset.py). One JSON object per line:

```json
{"case_id": "repo-t021-tests", "transcript": "transcripts/68734e48-....jsonl",
 "kind": "test_coverage", "commit": "d8e1221",
 "question": "Which test guards the extras declared in pyproject.toml against MissingExtraError.KNOWN_EXTRAS?",
 "answer": "tests/test_packaging.py, changed by commit d8e1221 when the assisted extra went.",
 "evidence": ["The `[assisted]` extra name disappears from `pyproject.toml` and `MissingExtraError.KNOWN_EXTRAS` (a packaging test enforces the pair)."]}
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

`transcripts/` holds the single exception: one excerpt of a real session in this
repository, kept verbatim — records untouched, harness bookkeeping (`mode`,
`permission-mode`, `file-history-snapshot`, `system`) left in beside the turns —
so the loader is exercised on the shape it meets in the wild while CI stays
deterministic. A gold record names a transcript; it is read from here when it is
the committed fixture and from the session directory otherwise.

## Invariants

`tests/evaluation/test_repo_row_kinds.py` checks the gold set: three kinds
covered, every case's transcript present, every evidence sentence found verbatim
in the document its case is answered from, every path an answer names present in
the tree. `tests/evaluation/test_repo_row_is_real.py` checks the corpus: one
committed fixture carrying the real record shape and its bookkeeping, every
answer's commit known to `git`, and an uncommitted transcript read from the
session directory.

Adding a case is a hand edit of `gold.jsonl`.
