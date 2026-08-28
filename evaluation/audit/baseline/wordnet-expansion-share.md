# WordNet-expansion REL-edge share — conv-30 (issue #141, AC3)

- Generated: `2026-08-28T22:36:00Z`
- Branch: `fix/issue-141` @ `9a3e9f7` (uncommitted: `graphknows/symbolic/ontology/senses.py` new,
  `lexical.py`/`skos.py` modified — the sense-anchored WordNet expansion this task measures)
- Command run (inside the `workspace` container, per Makefile/docker-compose):
  `python -m evaluation locomo --offset 152 --limit 81 --turns --reset`
  (matches `evaluation/locomo/cli.py --help` in this tree exactly; run with
  `logging.basicConfig(level=logging.INFO)` added ahead of `evaluation.__main__.main()`
  so `senses.py`'s `sense-anchored expansion: %r -> %s` INFO line would be visible)
- Database: `mem_eval_locomo_llm_free_conv_30` (namespace `eval_locomo_llm_free_conv-30`,
  dropped and re-created by `--reset` before this run)
- Container: `graphknows-workspace` / `graphknows-mcp` (image built from this worktree),
  `gk-issue-141-arcadedb-1` (ArcadeDB 26.5.1)

## The number

| Metric | Value |
|---|---|
| Total current REL edges (`is_current IS NULL OR is_current = true`) | **0** |
| `sense-anchored expansion: ... -> ...` INFO lines in the ingest log | **0** |
| REL edges attributable to WordNet-expanded labels | **0 / 0 — undefined (zero denominator)** |
| 22% baseline being compared against (issue #141) | 64 / 286 REL edges (first-sense mechanism, e.g. `Gina permits fashion` from "let") |
| Abstentions (mentions reaching the sense stage, rejected by threshold/margin) | **not observable** — `senses.py` logs a licensed expansion at INFO but has no log line for an abstained one (confirmed by reading `senses.sense_lemmas`, lines 90-120: both the `top1_sim < MIN_SENSE_SIM` and `top1_sim - top2_sim < MIN_SENSE_MARGIN` branches `return frozenset()` with no logging) |

**The share did not shrink from 22% to some smaller number — it collapsed to 0/0 because the
conv-30 graph itself has zero REL edges on this branch's current state.** This is a materially
different (and worse) result than what AC3 was written to measure, and the cause traces to a
real regression in this branch's code, not to the environment.

## What actually ran

The command executed cleanly end to end: ingest reported `turns=369 flush_chunks=0 entities=0
flush_nodes=0 errors=0 flush=142s` (`slowest: drain_turns=116s, resolve_entities=12s,
graph_embeddings=11s`), then all 81 questions were answered and judged (`RunReport` written,
exit code 0). No exception, no `flush failed` warning, no `INGEST FAILED` line appears anywhere
in the log. Direct queries against the database after the run confirm the summary line rather
than contradicting it:

```
SELECT count(*) FROM CHUNK          -> 0
SELECT count(*) FROM ENTITY         -> 0
SELECT count(*) FROM REL            -> 0
SELECT count(*) FROM FRAME_INSTANCE -> 0
```

(re-run with the audit `_REL_QUERY` from `evaluation/audit/cli.py` verbatim, against
`mem_eval_locomo_llm_free_conv_30`: `rel rows: 0`.)

A one-turn sanity probe in the same container, same models, same settings —
`Memory().ingest_memory(...)` + `flush_memory(...)` on `"Gina works at Acme Corp as an
engineer."` — came back `nodes=3, chunks=1, edges=0` with
`abstentions={'entities': 3, 'unfilled_core_unknown': 3, 'relations': 2}`: a small amount of
real graph, not nothing. So the pipeline is not globally broken in this environment; something
specific to the full 369-turn, 19-document conv-30 ingest zeroes out every extraction.

## Root cause: not this environment, a code regression on this branch

`python -m pytest tests/ontology tests/ingestion/test_lexical_labels.py
tests/ontology/test_sense_anchored_expansion.py -q` on this worktree (as already reported by
the implementation task) is not fully green:

```
FAILED tests/ontology/test_sense_anchored_expansion.py::test_wrong_sense_does_not_license_expansion
FAILED tests/ingestion/test_lexical_labels.py::TestDomainRangeSatisfaction::test_a_spoken_plumbing_predicate_is_still_offered
2 failed, 246 passed, 2 xfailed
```

The second failure is directly relevant to this measurement, and is stronger evidence than its
one-line summary suggests. `test_a_spoken_plumbing_predicate_is_still_offered` builds a scheme
whose property `requires` is reached by its own **literal, non-synonym label** ("The process
requires a signature." against a concept literally labelled `requires`) — no WordNet expansion
involved at all — and asserts `"requires" in relation_labels(...)`. It gets `[]`. If
`lexical.evoked()` (via `labels.py::relation_labels`/`class_labels`) is failing to return even a
direct, literal label match on real spaCy `Doc` objects, that is consistent with — and would
fully explain — a real 369-turn conversation producing zero class/property labels for every
chunk it processes, which is what the extraction pipeline (GLiNER-2, guided by those labels)
turned into zero entities and zero REL edges, while still spending real CPU time
(`drain_turns=116s`) doing so.

This was already flagged by an earlier task in this plan as "outside this task's files
(`labels.py`, untouched)" and left failing rather than silently changed, per instruction. This
report treats it as the most likely proximate cause of the 0-edge conv-30 graph, filed below as
a discovery — not fixed here, since this task's scope is measurement only and
`graphknows/` code is off-limits for it.

## Why this is not stubbed, faked, or approximated

Every number above came from the actual command, the actual container, and a live query against
the actual database — nothing here is a placeholder. The task's environment-defect exception
(no ArcadeDB / no corpus / no model) does not apply: ArcadeDB, the LoCoMo corpus, WordNet,
FrameNet and every baked model were all present and working (confirmed by the one-turn sanity
probe succeeding, and by all 81 questions answering and being judged). What is missing is a
non-empty graph, and its absence is code-shaped, not environment-shaped.

## What would need to happen to get the AC3 number

Once the `labels.py`/`evoked()` regression above is root-caused and fixed (tracked as a
discovery, not in scope here), re-run this exact command and re-read this file's "The number"
table — the total current REL edge count and the WordNet-expansion share will both change, and
whichever they land on should replace the "0 / 0" result recorded here rather than be appended
alongside it, so this file never carries two conflicting sets of numbers for the same measurement.
