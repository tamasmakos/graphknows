# Relation-vocabulary A/B — cco vs personal (conv-30)

- Generated: `2026-08-28T11:04:00Z`
- Command (both arms): `python -m evaluation locomo --offset 152 --limit 81 --turns --reset --namespace eval_relvocab_<arm>`
- Databases: `mem_eval_relvocab_cco_llm_free_conv_30`, `mem_eval_relvocab_personal_llm_free_conv_30`
- Reports: `evaluation/results/locomo/eval-llm_free-20260828080112-locomo.json` (cco), `evaluation/results/locomo/eval-llm_free-20260828083246-locomo.json` (personal)
- Audit sheets: `evaluation/audit/baseline/relvocab-cco-seed0.csv/.json`, `evaluation/audit/baseline/relvocab-personal-seed0.csv/.json` (drawn with `-n 150 --seed 0`)
- `GRAPHKNOWS_RELATION_ONTOLOGY` was confirmed to resolve to the intended asset for each token (`.../assets/cco/cco.json` for `cco`, `.../assets/personal/personal-profile.json` for `personal`) before the runs.

## The blocker is fixed — this is the real comparison

The previous attempt (see git history of this file) root-caused both arms
producing zero-REL-edge graphs to `graphknows/symbolic/ontology/skos.py::_wordnet_synonyms`
letting a `LookupError` (missing WordNet corpus) escape past its import-only
try/except and abort every window write. A prior task in this plan guarded the
lookup itself (a cached `_wordnet()` helper, mirroring `framenet.py::_framenet()`,
degrading to `set()` when the corpus cannot be made available). With that guard
in place and the WordNet corpus present in this container, **both arms ran to
completion, ingested a real graph, and produced comparable REL edges.**

### Sanity probe before committing to a full run

`Memory().ingest_memory(messages=..., session_id=...)` + `Memory().flush_memory(session_id)`
on 6 synthetic turns (personal profile): `nodes=8, chunks=2, turns=6, edges=0,
errors=[]`. CHUNK/ENTITY rows were gained and `flush_memory`'s `errors` list was
empty, so the ingest path was confirmed healthy before spending ~26 minutes per
arm on the full conv-30 run. This probe's `drain_turns` stage alone took
**271.5s** for one window — the FrameNet lexical-unit index build
(`framenet.py::_lus()`), confirmed below to scale to the full run rather than
hang.

### FrameNet cold-start cost (timed, not mistaken for a hang)

Each arm is a fresh `docker compose exec` — a new `graphknows-mcp` subprocess —
so the FrameNet LU index (per-process in-memory cache only) rebuilds once per
arm on the first window:

| Arm | `flush_elapsed_s` | `drain_turns` stage |
|---|---|---|
| cco | 1547.5s (~25.8 min) | 1531.1s (~25.5 min) |
| personal | 1554.2s (~25.9 min) | 1538.5s (~25.6 min) |

Both runs completed cleanly — ingest `errors=0` in both arms — confirming this
is the documented cold-start cost, not a second defect. (The rest of ingest —
`resolve_entities`, `graph_embeddings`, `pagerank`, `communities`, `topics` —
is a few seconds combined in both arms.)

## Result: both arms produced a real, comparable graph

| Metric | cco arm | personal arm |
|---|---|---|
| REL edges (current) | 356 | 356 |
| CHUNK / ENTITY-bearing vertices (`flush_nodes`) | 334 | 334 |
| `evidence_recall_at_probe` (mean) | 0.9759 | 0.9759 |
| `evidence_recall_in_context` (mean) | 0.9019 | 0.9019 |
| Accuracy overall (LLM judge) | 0.8519 | 0.8272 |
| Accuracy — `single_hop` | 0.8409 | 0.8636 |
| Accuracy — `temporal` | 0.8846 | 0.8077 |
| Accuracy — `knowledge_synthesis` | 0.8182 | 0.7273 |
| Abstention rate | 0.0617 | 0.0617 |
| Distinct predicates (audit sample) | 55 | 52 |
| Tier — `frame` (population) | 203 | 203 |
| Tier — `wordnet` (population) | 0 | 0 |
| Tier — `ontology` (population) | 105 | 144 |
| Tier — `lemma` (population) | 48 | 9 |
| Tier — `unclassified` (population) | 0 | 0 |

Top predicates (`predicate_population`, from the seed-0 audit manifest):

**cco** — `fn:Activity_ongoing:Activity->Agent` 48, `lemma:plans to` 42, `uses`
35, `environs` 22, `fn:Giving:Recipient->Theme` 14, `has sender` 13,
`fn:Giving:Donor->Theme` 12, `fn:Feeling:Emotion->Experiencer` 12, `interval is
before` 11, `fn:Assistance:Focal_entity->Helper` 8, `receives` 8.

**personal** — `fn:Activity_ongoing:Activity->Agent` 48, `likes` 45, `plansTo`
42, `worksFor` 21, `owns` 20, `fn:Giving:Recipient->Theme` 14, `attendee` 13,
`fn:Giving:Donor->Theme` 12, `fn:Feeling:Emotion->Experiencer` 12,
`fn:Assistance:Focal_entity->Helper` 8.

## Reading the numbers

**Evidence recall is identical to four decimals in both arms** —
`evidence_recall_at_probe=0.9759` / `evidence_recall_in_context=0.9019` — the
same saturation `evaluation/README.md` already documents for retrieval-metric
changes on this conversation: the system is generation-bound, so this pair of
metrics cannot discriminate between the two vocabularies here. This is
expected, not a null result specific to this A/B.

**Accuracy differs by 0.0247** (0.8519 cco vs 0.8272 personal), well inside the
harness's own noise band (~0.06, per the prior baseline notes) and with the
per-category deltas pointing in *both* directions (`single_hop` and `temporal`
trade places between arms) — consistent with judge/sampling noise on 81 cases,
not a systematic effect of the relation vocabulary. **Accuracy does not
distinguish the two arms.**

**The `fn:` (frame) tier is identical in both arms** (203/203, byte-for-byte
same predicate/count breakdown) — expected, since FrameNet projection does not
consult `GRAPHKNOWS_RELATION_ONTOLOGY` at all. Both arms drew the same 356
edges from the same underlying graph structure; only the labels attached to
the non-frame ~155 edges differ per vocabulary. This is the one axis in this
run that *does* separate the two arms:

- **cco** maps 105 edges into its ontology tier and leaves **48 edges
  (13.5% of all REL edges) as bare, unanchored `lemma:` predicates** (`plans
  to`, `is interest of`, `formerly worked for`, `teaches`, `is husband of`) —
  i.e. CCO's enterprise/commercial-frame vocabulary has no formal property for
  a sizeable share of what this personal, two-person conversation actually
  expresses.
- **personal** maps 144 edges into its ontology tier (`likes`, `plansTo`,
  `worksFor`, `owns`, `attendee`, `friendOf`) and leaves only **9 edges (2.5%)**
  unanchored. The 25-object-property personal vocabulary was hand-curated for
  exactly this kind of conversational content, and the predicate distribution
  bears that out: `likes`/`plansTo`/`worksFor`/`owns`/`attendee` are all
  legible, reusable relation names, where cco's mapped equivalents
  (`environs`, `interval is before`, `has sender`, `occurs at`, `receives`)
  read as generic/formal labels borrowed from a domain (commerce, logistics,
  time-interval algebra) this conversation isn't in.

Distinct-predicate counts are close (55 cco vs 52 personal) and not
informative on their own — the difference that matters is *which* tier the
edges land in, not how many distinct labels appear.

**Triplet-audit precision (`judge_span`/`judge_predicate`/`judge_direction`/
`judge_polarity_modality`) is not judged in this run** — both
`relvocab-cco-seed0.csv` and `relvocab-personal-seed0.csv` ship with the four
`judge_*` columns blank, per the same convention as the committed `#136`
baseline (`conv-30-seed0.csv`). A human judging pass over these two sheets
would be the strongest evidence for a quality delta and is not part of this
task; it remains open for whoever acts on this report next.

## Verdict

**The evidence favours the personal profile, on the one axis this run can
actually discriminate: predicate anchoring quality**, not on accuracy or
evidence recall (both read as noise/saturation between the two arms). The
personal vocabulary anchors 97.5% of non-frame-tier edges into a real ontology
property versus cco's 86.5%, and the anchored labels are legible for the kind
of personal, two-person conversational memory this vocabulary is meant to
serve — which is exactly the case issue #137 makes for demoting CCO to an
example and bundling the personal profile as the default. This is not a
blind/precision-scored result (the triplet audit is unjudged), so treat it as
directional evidence supporting the flip, not a statistically certified
precision win.

## Still true regardless of the above

The triplet-audit half of the decision (span / predicate / direction /
polarity_modality judgement) remains unjudged in both the existing `#136`
baseline sheet (`conv-30-seed0.csv`, `judged=0` on every tier) and the two new
sheets drawn here — a human judging pass is the natural next step before
calling the predicate-anchoring signal above a certified precision result
rather than a directional one.

## 2026-08-29 addendum: this A/B ran with the camelCase lexical surface broken

Issue #250 found that `skos.py::content_tokens` lowercased a label before
tokenizing it, destroying the only word-boundary signal a camelCase
schema.org-shaped label has — so 12 of the personal profile's 25 properties
(`worksFor`, `friendOf`, `lifePartnerOf`, `homeLocation`, `workLocation`,
`hasOccupation`, `knowsAbout`, `memberOf`, `alumniOf`, `relatedTo`,
`plansTo`, `lifeEvent`) and 1 of its 9 classes (`EducationalOrganization`)
had no lexical surface beyond their own unreadable concatenated label
(`worksfor`, `lifepartnerof`, `educationalorganization`, ...) at the time
this A/B ran (`2026-08-28T11:04:00Z`, per this file's own `Generated` line).
That bug was still present when the "personal 144 ontology-tier / 9
unanchored vs cco 105 / 48" predicate-anchoring result above was measured.

**This makes the personal arm's 144/9 result a floor, not a ceiling** — every
number above it was reachable with more than half the personal vocabulary's
properties effectively mute to normal conversational text (matchable only via
the literal camelCase string or embedding similarity, not via lexical
lookup). CCO has zero camelCase labels in either its classes or its 264
properties (confirmed in issue #250), so the cco arm's 105/48 is unaffected
by this bug and is not a floor in the same sense.

Re-running both arms with the fix in place is a separate ~52-minute two-arm
`evaluation locomo` docker run (matching this file's own cold-start timings:
~26 minutes per arm) and was not performed as part of this note — no number
in the table or prose above has been changed or re-measured here.
