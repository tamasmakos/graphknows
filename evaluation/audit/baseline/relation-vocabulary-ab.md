# Relation-vocabulary A/B — cco vs personal (conv-30)

- Generated: `2026-08-28T03:13:39Z`
- Command (both arms): `python -m evaluation locomo --offset 152 --limit 81 --turns --reset --namespace eval_relvocab_<arm>`
- Databases: `mem_eval_relvocab_cco_llm_free_conv_30`, `mem_eval_relvocab_personal_llm_free_conv_30`
- Reports: `evaluation/results/locomo/eval-llm_free-20260828025127-locomo.json` (cco), `evaluation/results/locomo/eval-llm_free-20260828030841-locomo.json` (personal)
- `GRAPHKNOWS_RELATION_ONTOLOGY` was confirmed to resolve to the intended asset for each token before the runs (`.../assets/cco/cco.json` for `cco`, `.../assets/personal/personal-profile.json` for `personal`), and the eval process's own `os.environ` (which `docker exec -e` sets) is spread wholesale into the spawned `graphknows-mcp` subprocess's environment (`graphknows/integrations/client/mcp.py`), so the variable reaches the ingest process in both arms.

## Result: the comparison could not be made

Both arms ran to completion (ingest + all 81 questions, ~14–17 minutes each) with no crash and no reported errors, but **both produced a graph with zero REL edges** — there is nothing to compare a relation vocabulary against.

`python -m evaluation audit sample --namespace eval_relvocab_<arm>_llm_free_conv_30 -n 150 --seed 0` refused to draw a sheet for either arm:

```
no current REL edges in mem_eval_relvocab_cco_llm_free_conv_30 — nothing to audit
no current REL edges in mem_eval_relvocab_personal_llm_free_conv_30 — nothing to audit
```

Direct queries against both databases confirm the same shape: `TURN` rows exist (369, the raw buffered conversation) but `CHUNK` and `ENTITY` are both `0`. No `relvocab-cco-seed0.csv/.json` or `relvocab-personal-seed0.csv/.json` are included in this commit — the audit tool did not write a manifest for either arm, and fabricating one from nothing would misrepresent a real number as measured. The four-tier census, top predicates and distinct-predicate count this task was to report are all vacuously zero/absent in both arms, for the same non-relation-vocabulary reason below.

## What blocked it

Root-caused with a scratch repro (`Memory().ingest_memory(messages=...)` + `Memory().flush_memory(...)`, 6–20 synthetic turns, reproduces at any scale, unrelated to the LoCoMo corpus or either relation vocabulary):

1. The Docker image is missing NLTK's `wordnet` corpus data (checked: not under `/opt/models/nltk_data`, `/home/graphknows/nltk_data`, or any of the other search paths). Something on the per-turn write path that a window emission triggers depends on it and raises `LookupError: Resource 'wordnet' not found` for *every* turn that closes a window (turn 3 and turn 6 of a 6-turn/stride-3 probe; in the real 369-turn conv-30 runs this is every `_chunk_stride`'th turn plus the final one).
2. That exception is swallowed three layers up without a trace: `STMService.ingest_messages` (`evaluation`-independent, `graphknows/ingestion/stm/service.py`) catches it per-turn into a local `errors: list[str]`, but `Memory._ingest_buffered_turns` (`graphknows/memory.py`) only reads `result.chunk_count` and `result.abstentions` off the returned `IngestResult` — it never looks at `result.errors`, so the failure vanishes. `flush_memory()` then reports `chunks: 0`, `abstentions: {}`, `errors: []` — a graph that failed on every write is indistinguishable from a conversation that legitimately had nothing to extract.

Net effect: `--turns` ingestion (the mode both arms were required to use, per the issue's evidence_recall@probe rationale) is currently non-functional in this environment for either relation vocabulary, and the failure is invisible in every counter the eval harness surfaces — which is also why the accuracy/evidence-recall numbers below read as floor-with-abstention rather than a real signal.

An attempt was made to unblock the measurement by installing the missing corpus at runtime (`nltk.download('wordnet', download_dir='/opt/models/nltk_data')`, then unzipping — the downloader fetches the zip but does not extract it there). That fixed the `LookupError`, but the same 6-turn repro then ran for **over 10 minutes of continuous 99%-CPU work with no completion** (previously ~40s when the lookup failed fast). Whatever wordnet-dependent code executes once the corpus is actually present appears to scan or iterate the corpus in a way that does not scale even to 6 turns — a second, independent defect. The run was killed (container restart) rather than left to find out how long it would take; the corpus was not made a permanent fix, and no further attempt was made to re-run the two arms with it installed.

## Numbers observed (not a vocabulary comparison — both arms hit the same defect)

Recorded for completeness since the runs did execute; read strictly as "both arms produced an empty graph," not as a personal-vs-cco result.

| Metric | cco arm | personal arm |
|---|---|---|
| REL edges (current) | 0 | 0 |
| CHUNK / ENTITY vertices | 0 / 0 | 0 / 0 |
| `evidence_recall_at_probe` (mean) | 0.000 | 0.000 |
| `evidence_recall_in_context` (mean) | 0.000 | 0.000 |
| Accuracy overall (judge, confirmation only — harness noise ~0.06) | 0.000 | 0.000 |
| Accuracy — `single_hop` | 0.000 | 0.000 |
| Accuracy — `temporal` | 0.000 | 0.000 |
| Accuracy — `knowledge_synthesis` | 0.000 | 0.000 |
| Abstention rate | 0.9506 | ~0.95 (81 cases, near-all `'Unknown'`) |

No number here favours either arm — they are identical because neither arm has a graph to retrieve from. **No conclusion about the bundled default is drawn from this run.**

## Still true regardless of the above

The triplet-audit half of the decision (span / predicate / direction / polarity_modality judgement) remains unjudged in the existing `#136` baseline sheet (`conv-30-seed0.csv`, `judged=0` on every tier) — this run neither advances nor blocks that half, since no new sheets could be drawn either.
