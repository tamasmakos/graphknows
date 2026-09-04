# Pre-swap baseline — `llm_assisted` @ `3fc9a47`

- Commit: `3fc9a47ac64283b5adbdb92a00a42cf3a3600529`
- Corpus slice: locomo conv-30, questions 152-171 (--offset 152 --limit 20), 19 sessions ingested
- Mode: `llm_assisted`  Configuration: `all-on`  Seed: `0`
- Graph: `mem_decoder_swap_llm_assisted_conv_30`
- Run: `eval-llm_assisted-20260904055423`  (2026-09-04T09:07:12.285873+00:00)

Recorded for FR-030, before the decoder swap. Evidence recall and the four audit
axes are the gate; headline accuracy and ingest wall-clock are confirmation only.

## Gate metrics

| Metric | Value |
|---|---|
| evidence recall @probe | 1.000 |
| evidence recall in context | 1.000 |
| audit precision — span | 0.727 (109/150 judged) |
| audit precision — predicate | 0.820 (123/150 judged) |
| audit precision — direction | 0.833 (125/150 judged) |
| audit precision — polarity_modality | 0.980 (147/150 judged) |
| audit precision — all_four | 0.587 (88/150 judged) |

## Confirmation only

| Metric | Value |
|---|---|
| headline accuracy (LLM judge) | 1.000 over 20 cases |
| mean judge score | 0.945 |
| abstentions | 0 / 20 |
| ingest wall-clock | 7551.7s (buffer 7436.6s + flush 115.1s) |
| ingest cost | $0.0000 |
| query cost | $0.0410 |

Accuracy by category: `knowledge_synthesis` 1.000, `single_hop` 1.000, `temporal` 1.000

## Graph audited

| Tier | population | sampled | span | predicate | direction | polarity_modality |
|---|---|---|---|---|---|---|
| `frame` | 64 | 50 | 0.360 | 0.640 | 0.780 | 0.960 |
| `wordnet` | 0 | 0 | — | — | — | — |
| `ontology` | 69 | 50 | 0.880 | 0.880 | 0.980 | 0.980 |
| `lemma` | 176 | 50 | 0.940 | 0.940 | 0.740 | 1.000 |
| `unclassified` | 0 | 0 | — | — | — | — |

Judged sheet: `decoder-swap-3fc9a47-assisted-sheet.csv` (150 rows).
Judge: Claude Opus 5 acting as the audit's human-in-the-loop judge, not a human. Verdicts are checked in beside this report so they can be re-judged.

## Reproducing this

```sh
python -m evaluation --compare --limit 20 --offset 152 --namespace decoder_swap
python -m evaluation audit sample --namespace decoder_swap_llm_assisted_conv-30 -n 150 --seed 0
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
- Tiers larger than the per-tier quota were sampled, not censused, so the Wilson intervals carry sampling error as well as judging uncertainty.
