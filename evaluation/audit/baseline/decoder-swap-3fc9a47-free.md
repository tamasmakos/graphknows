# Pre-swap baseline — `llm_free` @ `3fc9a47`

- Commit: `3fc9a47ac64283b5adbdb92a00a42cf3a3600529`
- Corpus slice: locomo conv-30, questions 152-171 (--offset 152 --limit 20), 19 sessions ingested
- Mode: `llm_free`  Configuration: `all-on`  Seed: `0`
- Graph: `mem_decoder_swap_llm_free_conv_30`
- Run: `eval-llm_free-20260904055423`  (2026-09-04T06:55:04.868587+00:00)

Recorded for FR-030, before the decoder swap. Evidence recall and the four audit
axes are the gate; headline accuracy and ingest wall-clock are confirmation only.

## Gate metrics

| Metric | Value |
|---|---|
| evidence recall @probe | 1.000 |
| evidence recall in context | 1.000 |
| audit precision — span | 0.634 (83/131 judged) |
| audit precision — predicate | 0.779 (102/131 judged) |
| audit precision — direction | 0.931 (122/131 judged) |
| audit precision — polarity_modality | 0.977 (128/131 judged) |
| audit precision — all_four | 0.534 (70/131 judged) |

## Confirmation only

| Metric | Value |
|---|---|
| headline accuracy (LLM judge) | 0.850 over 20 cases |
| mean judge score | 0.830 |
| abstentions | 1 / 20 |
| ingest wall-clock | 3368.6s (buffer 3296.2s + flush 72.4s) |
| ingest cost | $0.0000 |
| query cost | $0.0444 |

Accuracy by category: `knowledge_synthesis` 0.800, `single_hop` 1.000, `temporal` 0.846

## Graph audited

| Tier | population | sampled | span | predicate | direction | polarity_modality |
|---|---|---|---|---|---|---|
| `frame` | 62 | 62 | 0.371 | 0.694 | 0.871 | 0.968 |
| `wordnet` | 0 | 0 | — | — | — | — |
| `ontology` | 69 | 69 | 0.870 | 0.855 | 0.986 | 0.986 |
| `lemma` | 0 | 0 | — | — | — | — |
| `unclassified` | 0 | 0 | — | — | — | — |

Judged sheet: `decoder-swap-3fc9a47-free-sheet.csv` (131 rows).
Judge: Claude Opus 5 acting as the audit's human-in-the-loop judge, not a human. Verdicts are checked in beside this report so they can be re-judged.

## Reproducing this

```sh
python -m evaluation --compare --limit 20 --offset 152 --namespace decoder_swap
python -m evaluation audit sample --namespace decoder_swap_llm_free_conv-30 -n 150 --seed 0
python -m evaluation audit report --sheet <judged sheet>
```

## Read this before quoting a number

- The harness is unmodified (FR-029); only the reports below are checked in.
- --offset 152 is what pins the slice to conv-30: offset 0 is conv-26.
- A private namespace (decoder_swap) keeps this run from over-writing the graph the older evaluation/audit/baseline/conv-30-seed0* reports were drawn from.
- evidence_recall is 1.000 on both arms because the document harness makes every chunk a multi-turn transcript; --turns is what restores headroom. As a gate at this scale it can only detect a regression, never an improvement.
- Judging standard, so a re-judge lands in the same place: `span` is 1 only when BOTH endpoints name something the evidence actually says (a projected frame endpoint must be the right surface; a minted event node must denote an event that is really there); `predicate` asks whether the relation label describes what the evidence asserts between those two; `direction` asks whether head and tail are the right way round for that label; `polarity_modality` asks whether the recorded assertion status (blank = plainly asserted) is right.
- The two arms share the frame and ontology tiers verbatim, so their judgements were carried across rather than re-judged, and only the assisted arm's new `lemma:` tier was judged fresh.
- `audit_precision` pools every judged row, and the stratified draw is not population-weighted, so the two arms' pooled numbers are NOT directly comparable: the assisted arm's sample is one third `lemma:`, a tier the llm_free graph does not have at all. Compare the per-tier table, tier by tier.
- ingest_cost_usd is 0.0 on both arms because the harness only counts the topic-summarisation spend; the assisted arm's per-chunk relation calls are not metered there. Read the wall-clock, not the cost, for the assisted arm.
- Every tier was judged in full (population == sampled), so the Wilson intervals are judging uncertainty, not sampling error.
