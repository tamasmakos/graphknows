# Stratified Triplet Audit

- Sheet: `conv-30-seed0.csv`
- Database: `mem_eval_locomo_llm_free_conv_30`
- Seed: `0`
- Generated: `2026-08-28T01:17:09.048420+00:00`

Precision measured directly on the graph — no retrieval in the loop.
Each cell is `precision [95% Wilson interval]` over the rows judged on that axis.

## Precision by tier

| Tier | population | sampled | no evidence | judged | span | predicate | direction | polarity_modality | all_four |
|------|------------|---------|-------------|--------|---|---|---|---|---|
| `frame` | 2551 | 51 | 0 | 0 | — | — | — | — | — |
| `wordnet` | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `ontology` | 105 | 51 | 0 | 0 | — | — | — | — | — |
| `lemma` | 48 | 48 | 0 | 0 | — | — | — | — | — |
| `unclassified` | 0 | 0 | 0 | 0 | — | — | — | — | — |

## verifier_score by tier

The continuous monitor between audits: computed at flush and stored on every REL edge. A NULL means the verifier never scored that edge — deliberately distinguishable from a scored 0.0.

| Tier | scored | NULL | mean | median |
|------|--------|------|------|--------|
| `frame` | 0 | 51 | — | — |
| `wordnet` | 0 | 0 | — | — |
| `ontology` | 51 | 0 | 0.618 | 0.598 |
| `lemma` | 48 | 0 | 0.561 | 0.551 |
| `unclassified` | 0 | 0 | — | — |

## Read this before quoting a number

- verifier_score coverage is PARTIAL: scored on `ontology`, `lemma`, never scored on `frame`. Read no mean below as a whole-graph number.
- `frame`: verifier_score is NULL on every sampled edge - `RelationVerifierGate` runs only in the relex write path, so it structurally cannot score this tier. Its mean is not a health number here.
- `frame`: 51 rows sampled, none judged yet.
- `wordnet`: no edges of this tier in the graph — nothing to audit.
- `ontology`: 51 rows sampled, none judged yet.
- `lemma`: 48 rows sampled, none judged yet.
- `unclassified`: no edges of this tier in the graph — nothing to audit.
