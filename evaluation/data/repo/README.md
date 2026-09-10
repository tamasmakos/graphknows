# Repository-transcript gold set (FR-044)

`gold.jsonl` is the hand-written gold set for the repository row, loaded by
[`evaluation/repo/dataset.py`](../../repo/dataset.py). One JSON object per line:

```json
{"case_id": "repo-t054-decision", "transcript": "transcripts/t054-repo-row.jsonl",
 "kind": "decision_location", "question": "Where does the repository-transcript row load its ingest units?",
 "answer": "In evaluation/repo/dataset.py, in load_units.",
 "evidence": ["Settled: the loading lives in evaluation/repo/dataset.py, where load_units returns one IngestUnit per transcript."]}
```

`evidence` holds the transcript sentences the answer is read off, verbatim, so
evidence recall is scored against the same text that was ingested.

## Kinds

Three, one question each per transcript:

- `decision_location` — where a thing was settled, not everywhere it was discussed;
- `change_rationale` — the reason actually given at the time;
- `test_coverage` — which test guards a symbol.

## Corpus

`transcripts/` holds agent-harness JSONL for three pieces of work in this
repository — the transcript checkpoint (T050), the repository row itself (T054)
and the hand-labelled identity sample. Each file is an excerpt: the records that
carry the decision, kept in the harness shape
[`graphknows.ingestion.parsers.transcript`](../../../graphknows/ingestion/parsers/transcript.py)
reads, with the session's bookkeeping records left in so the loader is exercised
on the shape it will meet in the wild.

Answers are cross-checked against the tree they describe — the file, symbol and
test each answer names exists at the commit that added it.

Adding a case is a hand edit of these two files. Invariants are checked by
`tests/evaluation/test_repo_row_kinds.py`: three kinds covered, every case's
transcript present, and every evidence sentence found verbatim in the document
its case is answered from.
