# Relation-vocabulary lexical-surface count — personal-profile digest (issue #250)

- Generated: `2026-08-29T15:13:59Z`
- Branch: `fix/issue-250` (uncommitted: `graphknows/symbolic/ontology/skos.py`'s
  `content_tokens` camelCase fix, `tests/ontology/test_skos_lexicalization.py`
  — the fix this task measures)
- Command run (inside the `workspace` container, per Makefile/docker-compose,
  image `gk-issue-250-workspace`):
  `docker run --rm -v "$(pwd):/app" -w /app gk-issue-250-workspace python measure_relation_surface.py`
  where `measure_relation_surface.py` calls `skos.build_scheme()` twice over
  `graphknows/symbolic/ontology/assets/personal/personal-profile.json` — once
  with the tree's current `content_tokens`, once with `skos.content_tokens`
  monkeypatched to the pre-fix two-line tokenizer (`re.findall(r"[A-Za-z]+",
  label.lower())`, i.e. lowercase-then-tokenize) — and for every concept counts
  `len(concept.alt_labels - {concept.pref_label.lower()})`: how many usable
  surfaces a concept has beyond its own raw label. The pre-fix tree was never
  checked out or reverted; the old tokenizer is reproduced inline in the
  measurement script only, per the task's instruction.
- Digest: `graphknows/symbolic/ontology/assets/personal/personal-profile.json`
  — 25 properties, 9 classes.

## The numbers

| | properties w/ zero alt_labels beyond prefLabel | classes w/ zero alt_labels beyond prefLabel |
|---|---|---|
| **Before** (pre-fix tokenizer) | 19 / 25 | 9 / 9 |
| **After** (current tree) | 7 / 25 | 8 / 9 |
| **Delta** | 12 properties gained a surface | 1 class gained a surface |

Before, zero-surface properties (19): `about`, `alumniOf`, `attendee`,
`colleague`, `friendOf`, `hasOccupation`, `homeLocation`, `knowsAbout`,
`lifeEvent`, `lifePartnerOf`, `location`, `memberOf`, `nationality`, `parent`,
`plansTo`, `relatedTo`, `spouse`, `workLocation`, `worksFor`.

After, zero-surface properties (7): `about`, `attendee`, `colleague`,
`location`, `nationality`, `parent`, `spouse`.

Before, zero-surface classes (9, all of them): `Country`,
`EducationalOrganization`, `Event`, `Occupation`, `Organization`, `Person`,
`Place`, `Product`, `Thing`.

After, zero-surface classes (8): `Country`, `Event`, `Occupation`,
`Organization`, `Person`, `Place`, `Product`, `Thing` (`EducationalOrganization`
dropped out — it now splits into `educational` and `organization`).

## Reading the numbers

**The fix moved exactly 12 of 25 properties and 1 of 9 classes from "zero
alt_labels beyond prefLabel" to "at least one extra token", matching the
ticket's estimate — but "gained an alt_label" is not the same claim as
"gained a REACHABLE surface", and one of the 12 does not clear that second
bar.** `worksFor` -> `works`/`for` is filtered by `_STRUCTURAL`/length, so its
real gain is `worksfor` (junk) replaced by no new token — but properties like
`friendOf` -> `friend`, `lifePartnerOf` -> `life partner`,
`homeLocation`/`workLocation` -> `home`/`work`/`location`, and
`EducationalOrganization` -> `educational organization` go from one
unreachable concatenated string (`lifepartnerof`, `educationalorganization`)
to tokens that actually occur in conversational text and are reachable
through `lexical.evoked`'s STRONG rule (single-token labels) or WEAK branch.
`hasOccupation` needed one more step: `content_tokens` alone still split it
into `has`/`occupation`, and `has` — a spaCy stopword, absent from
`lexical.lemma_view`'s output — meant STRONG could never see all of the
label's tokens present, and the WEAK branch's single-token gate
(`len(content_tokens(label)) != 1`) excluded it too, so the concept was
alt_label-nonempty but match-dead. `_STRUCTURAL` now also strips `has` (it
sits with the other function words already there: `of`, `to`, `for`, ...),
which collapses `hasOccupation` to the single token `occupation` — both
STRONG- and WEAK-reachable — and, as a side effect, fixes the same
`has <noun>` shape across every "has X" family/part relation CCO ships
(`has brother` -> `brother`, `has mother` -> `mother`, ~70 properties total),
not only this one label. This is cause 1 of #250, confirmed fixed for 11 of
the 12 properties.

`knowsAbout` is the exception, and stays broken: it tokenizes to
`knows`/`about`, `about` is also a stopword `lemma_view` never emits, and
unlike `has` it cannot be added to `_STRUCTURAL` — CCO's own `about` /
`is about` property (present in this same digest) IS the single word
"about", and structuralizing it would delete that property's only reachable
surface as collateral damage. `knowsAbout`'s alt_labels are non-empty
(`knows`, `about`, `knows about`), which is why it does not appear in either
zero-surface list below, but STRONG requires both tokens present and WEAK
requires exactly one, so no path in `lexical.evoked` can ever select it. Left
unfixed and reported here rather than silently claimed as working.

**The 7 properties still at zero (`spouse`, `parent`, `colleague`,
`nationality`, `attendee`, `location`, `about`) are NOT the defect #250
reported.** Each is already one ordinary English word — the label IS the
content word, so `content_tokens` (before or after the fix) returns exactly
that one word back, and "beyond the label" is correctly empty. These reach
text via `lexical.evoked`'s STRONG all-tokens-present rule and via
`lexical._sense_anchored_matches`'s single-token branch
(`len(content_tokens(label)) != 1` gate lets a one-word label through
directly), not through any altLabel expansion. The 20/25 headline in #250
conflated "camelCase mush" (12 properties, the real bug) with "single word,
nothing to add" (7 properties here, working as designed) — the true
camelCase defect was 12/25, not 20/25.

The task's own residual list also named `sibling` as a same-shape single-word
case; the measured `alt_labels` for `sibling` is `{'sibling', 'sible'}`, not
`{'sibling'}`, because `lexicalize`'s spaCy lemmatisation step
(`_lemmatise`, property-only) mislemmatises the gerund-shaped noun "sibling"
to "sible". That extra entry is spaCy lemmatizer noise unrelated to #250 (it
predates this fix and is not camelCase-shaped), so `sibling` is excluded from
the zero-surface count above on a technicality, not because it has a useful
second surface — it still reaches text the same way as the other six, via its
own label as a single content word.

**Cause 2 as originally reported (`_wordnet_synonyms` hardcoding
`wn.VERB`) required no change and was not touched.** That function does not
exist in this tree: issue #141 deleted build-time WordNet synonym expansion
from `lexicalize` entirely (see `skos.py`'s "WHAT IS DELIBERATELY NOT DONE"
docstring section and
`tests/ontology/test_skos_lexicalization.py::TestMissingWordnetCorpus`).
Synonym expansion is now a match-time decision in
`lexical._sense_anchored_matches`, which takes its part of speech from the
mention rather than a hardcoded guess (`_candidate_mentions` returns `"n"` or
`"v"` per token) and gates properties with `pos_gate = {"n", "v"}` — nouns are
already allowed. Noun-valued properties such as `spouse`, `sibling`,
`colleague`, `nationality`, and `location` are reachable as nouns today; the
hardcoded-verb defect #250 described for cause 2 died with #141, before this
task started.
