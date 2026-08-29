# Vocabulary-profile derivation — before/after the hardcoded label lists (issue #251)

- Generated: `2026-08-29`
- Branch: `fix/issue-251` (uncommitted: `graphknows/symbolic/ontology/skos.py`'s
  `VocabularyProfile`/`vocabulary_profile`/`_upper_class_labels`/
  `_frequent_tokens` — the change this note measures — plus
  `tests/ontology/test_vocabulary_profile.py`)
- Command run (this tree's Python, `graphknows` importable from repo root, no
  container needed — the measurement only calls `graphknows.symbolic.ontology.skos`
  functions, no spaCy/WordNet/network path): a throwaway script,
  `measure_vocab_profile.py`, run from the worktree root and deleted afterward
  (not committed; not part of the package). It calls
  `skos.vocabulary_profile(digest)` and `skos._class_depths_and_descendant_counts`
  on the parsed JSON of each digest below, and reproduces the OLD rule inline
  (`_BFO_ABSTRACT`/`_STRUCTURAL`, copied verbatim from `git show HEAD` before
  this branch's change replaced them) rather than importing it, since HEAD's
  `skos.py` no longer defines it.
- Digests: `graphknows/symbolic/ontology/assets/cco/cco.json` (1437 classes,
  264 properties), `graphknows/symbolic/ontology/assets/personal/personal-profile.json`
  (9 classes, 25 properties).
- Falsifiable by: re-running `measure_vocab_profile.py` (reconstructable from
  this note — it is three calls: `vocabulary_profile(digest)`,
  `_class_depths_and_descendant_counts(classes)`, and the OLD `_BFO_ABSTRACT`
  membership test applied to each property's `domain`/`range`) against the same
  two digest files; the digests are content-hashed inputs, not something this
  note interprets.

## Classes flagged upper-ontology, with depth and descendant count

**CCO** (13 of 1437 classes, `depth<=3` and `descendants>=8 or >=5% of 1437≈72`):

| class | depth | descendants |
|---|---|---|
| entity | 0 | 1436 |
| continuant | 1 | 1063 |
| independent continuant | 2 | 613 |
| material entity | 3 | 546 |
| occurrent | 1 | 371 |
| specifically dependent continuant | 2 | 292 |
| process | 2 | 253 |
| realizable entity | 3 | 215 |
| generically dependent continuant | 2 | 154 |
| Act | 3 | 154 |
| Information Content Entity | 3 | 153 |
| quality | 3 | 75 |
| Process Profile | 2 | 74 |

**personal profile** (1 of 9 classes, floor is `max(8, 0.05*9=0.45)=8`):

| class | depth | descendants |
|---|---|---|
| Thing | 0 | 8 |

`Organization` and `Place` each have 1 descendant, `Country`/`Event`/
`Occupation`/`Product` have 0 — all below the 8-descendant floor, so none
qualify. This matches the issue's stated measurement (`Thing` flagged,
`Organization`/`Place` not).

## Object properties flagged upper-ontology: before vs after

| | CCO | personal |
|---|---|---|
| OLD (`_BFO_ABSTRACT`, hardcoded) | 132 | 0 |
| NEW (derived `upper_classes`) | 105 | 5 |
| lost (OLD only) | 27 | 0 |
| gained (NEW only) | 0 | 5 |
| kept (both) | 105 | 0 |

**CCO — the 27 lost** (flagged by the hardcoded BFO list, not flagged by the
derived rule): `aggregate has capability`, `aggregate has disposition`,
`aggregate has role`, `capability of aggregate`, `coincides with`, `connected
with`, `disconnected with`, `disposition of aggregate`, `externally connects
with`, `first instant of`, `has first instant`, `has last instant`, `has
nontangential part`, `has organizational context`, `has spatial part`, `has
subordinate role`, `has tangential part`, `is organizational context of`, `is
subordinate role to`, `last instant of`, `nontangential part of`, `partially
overlaps with`, `role of aggregate`, `spatial part of`, `spatially projects
onto`, `tangential part of`, `temporally projects onto`.

All 27 are typed against `object aggregate` (depth 4, 24 descendants),
`spatial region` (depth 4, 26), `spatiotemporal region` (depth 2, 0),
`temporal region` (depth 2, 37), `role` (depth 4, 21), `disposition` (depth 4,
185), or `function` (depth 5, 138) — every one of the seven either sits one or
more hops past the `depth<=3` cutoff, or (the two temporal/spatial-region
leaves) is inside the depth cutoff but well under the 72-descendant floor.
`disposition`'s 185 descendants is the closest near-miss — well past the
descendant floor but one hop too deep. **The derivation does not reproduce
these 27
— stated plainly, not rounded away.** They matter less than the count alone
suggests: `permits`/`requires`/`realizes`/`is required by` (the class the
issue measured at 64 of 286 REL edges) are all typed on `process`,
`process regulation`'s domain/range chain, or `realizable entity` directly,
which do clear the cutoff (see next section) — the 27 lost are a second,
smaller family (fiat-part/aggregate mereology) the old list also happened to
sweep in, not part of the measured 64-edge failure.

**CCO — 0 gained, 0 false positives.** Every one of the 105 newly-derived
flags is a subset of the old 132; the derived rule never flags a property the
hardcoded list didn't already flag.

**personal — the 5 gained** (flagged by the derived rule, not by the
hardcoded CCO/BFO list, because the hardcoded list is CCO-spelled and none of
its 23 tokens collide with this vocabulary's 9 class labels): `about`,
`dislikes`, `knowsAbout`, `likes`, `plansTo` — all domain- or range-typed on
`Thing`, this digest's own upper class. This is issue #251's central claim
made concrete: the old rule is inert here (0 flagged either way from CCO's
list), the derived rule protects it the same way it protects CCO.

## The four probe properties

| property | OLD | NEW |
|---|---|---|
| `permits` | flagged | flagged |
| `requires` | flagged | flagged |
| `realizes` | flagged | flagged |
| `is required by` | flagged | flagged |
| `owns` | not flagged | not flagged |
| `works for` | not flagged | not flagged |
| `teaches` | not flagged | not flagged |
| `lives in` | not flagged | not flagged |

All eight readings confirmed exactly as the issue requires: the four
vacuous-constraint properties that produced 64 of 286 REL edges (22%) on
conv-30 stay flagged under the derived rule, and the four everyday personal
properties stay unflagged in both.

## Personal-profile properties newly suppressed structurally — lexical half unaffected

`about`, `dislikes`, `knowsAbout`, `likes`, `plansTo` lose only their
domain/range-satisfiability route into `relation_labels`
(`graphknows/symbolic/ontology/labels.py::relation_labels` — `concept in
lexical or _upper_ontology_property(concept)` gates the structural half only;
`_upper_ontology_property` is never consulted by `evoked`/
`_sense_anchored_matches`, which build the lexical half from `concept.tokens`/
`concept.alt_labels`). Each property's own name is still a lexical trigger:
text that says "likes" or "plans to" still surfaces the property through the
lexical-evocation path exactly as before this change — verified by reading
`relation_labels`, not re-run here, since `lexicalize`/`content_tokens` for
these five concepts is untouched by `vocabulary_profile` (their labels are not
upper-class labels and are not in `structural_tokens`, so none of their own
tokens are stripped).

## Structural tokens derived, against the old 36-item `_STRUCTURAL` list

(The task text says "the 35" — the actual list at `git show HEAD` has 36
entries, counted directly; noted rather than silently corrected to match.)

**CCO** — derived `structural_tokens` (18): `act`, `artifact`, `content`,
`continuant`, `dependent`, `entity`, `function`, `generically`, `has`,
`independent`, `information`, `material`, `occurrent`, `process`, `profile`,
`quality`, `realizable`, `specifically`.

- Re-derived from the old list (13): `act`, `artifact`, `content`,
  `continuant`, `dependent`, `entity`, `generically`, `has`, `information`,
  `occurrent`, `process`, `profile`, `quality`.
- Not re-derived (23): `a`, `acts`, `an`, `and`, `artifacts`, `at`, `by`,
  `entities`, `for`, `from`, `in`, `of`, `on`, `ontology`, `or`, `processes`,
  `profiles`, `qualities`, `role`, `roles`, `the`, `to`, `with`.
- Why the gap doesn't matter, measured (not assumed): `a`/`an`/`at`/`by`/`in`/
  `of`/`on`/`or`/`to` are already dropped by `content_tokens`'s
  `len(w) > 2` filter regardless of any structural set — the module's own
  docstring says exactly this. `and`/`for`/`from`/`the`/`with` are covered by
  the explicit `_FUNCTION_WORDS` list kept for this reason (English grammar,
  not ontology scaffolding) — and measured directly: none of `and`/`the`/
  `for`/`with`/`from` occurs as a `content_tokens` token in any CCO class or
  property label at all (0/1437, 0/264 document frequency each), so they
  wouldn't be "re-derived" by the 10%-document-frequency rule even without
  `_FUNCTION_WORDS` catching them first — CCO's own labels simply don't use
  these words. The four plurals `acts`/`entities`/`artifacts`/`processes`/
  `profiles`/`qualities`/`roles` (7 tokens) never occur either — CCO's
  singular label vocabulary (`act`, `entity`, ...) is what repeats, not the
  plural, so their singulars are re-derived and the plurals correctly are not.
  `ontology` occurs in 0 labels (it was defensive scaffolding in the old list,
  never load-bearing). `role`/`roles` is the one real miss: `role` has 1.9%
  class-label and 1.5% property-label document frequency — present but under
  the 10% cutoff, so it is not derived, and is not an upper-class label either
  (depth 4, 21 descendants, outside both the depth<=3 cutoff and the
  descendant floor) — consistent with the 27-property gap in the previous
  section (`role of aggregate`, `has subordinate role`, etc.), not a separate
  defect.

**personal profile** — derived `structural_tokens` (10): `country`,
`educational`, `event`, `location`, `occupation`, `organization`, `person`,
`place`, `product`, `thing`. **0 of the old 36-item CCO list re-derive here**
— every one of the personal profile's structural tokens is a token from its
own 9 class labels (`Thing`, `Organization`, `Place`, `Country`, `Event`,
`Occupation`, `Product`, `EducationalOrganization` splitting into
`educational`/`organization`); none of the old CCO-spelled tokens (`act`,
`continuant`, `entity`, ...) appear in this vocabulary's labels at all. This
is the same point issue #251 opens with: the old list was CCO-spelled and
inert elsewhere by construction, while the derived one is vocabulary-specific
by construction.

## Summary

The derivation reproduces the issue's headline numbers exactly: 105 of the
132 old-flagged CCO properties (0 false positives), all four probe properties
correctly on each side of the line, and the personal profile's `Thing`-ranged
properties (`about`/`dislikes`/`knowsAbout`/`likes`/`plansTo`) newly protected
with their lexical surface untouched. What it does **not** reproduce is the
27-property mereotopology/role/disposition family CCO's hardcoded
`_BFO_ABSTRACT` list also caught (`spatial part of`, `has subordinate role`,
`coincides with`, ...) — those classes sit one hop deeper (`role`,
`disposition`, `function` under `realizable entity`) or on a smaller branch
(`object aggregate`, the three region classes) than the 3-hop/8-descendant
cutoff reaches. They are a separate, smaller failure mode than the one the
issue measured (64/286 edges from `permits`/`requires`/`realizes`), not part
of it, but they are a real coverage loss and are recorded here rather than
rounded away.
