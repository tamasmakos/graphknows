# Entity-typing vocabulary census — the seven COARSE types (#154)

- Generated: `2026-08-29T05:57:30+02:00`
- Method: offline, no container, no eval run. Loaded the bundled digests
  directly with the repo's own scheme-building functions
  (`graphknows.symbolic.ontology.skos.build_scheme` /
  `extend_scheme` / `lexical_index` / `coarse_aliases`) and ran the real
  `graphknows.ingestion.stm.ingest._lexical_candidates` against each of the
  seven coarse types (`graphknows.symbolic.ontology.labels.COARSE`), once per
  vocabulary below. "Before the fix" reproduces the pre-fix candidate set by
  taking the SKOS altLabel index as-is, with no curated-alias check and no
  content-token subset filter — the behaviour `_lexical_candidates` had before
  the sibling change in this plan.
- Vocabularies:
  - **CCO** — `graphknows/symbolic/ontology/assets/cco/cco.json` (1437
    classes; 1443 after the overlay below adds its own concepts).
  - **Overlay** — `graphknows/symbolic/ontology/assets/cco/conversational.json`,
    always merged on top of whichever base ontology is loaded
    (`GraphKnowsSettings.overlay_source` is independent of
    `GRAPHKNOWS_ONTOLOGY` — see `load_scheme`). Its `coarseAliases` block is
    the curated fix.
  - **Personal profile** — `graphknows/symbolic/ontology/assets/personal/personal-profile.json`
    (9 classes: Thing, Person, Organization, EducationalOrganization, Place,
    Country, Event, Occupation, Product), loaded as `GRAPHKNOWS_ONTOLOGY` the
    same way CCO is, and merged with the same bundled overlay.

## Before the fix: CCO, fragment anchoring

Every content word of a class label became a standalone `altLabel`
(`skos.lexicalize`), so a coarse type anchored on any class that merely
*contains* that word — with no ranking check preventing a single-candidate set
from winning outright at score 1.0.

| Coarse type | Candidates (`lex_index` hit, unfiltered) |
| --- | --- |
| `person` | `Person`, `Neutral Person`, `Allied Person`, `Enemy Person` |
| `organization` | `Organization`, `Government Organization`, `Civil Organization`, `Incorporated Organization`, `Geopolitical Organization`, `Commercial Organization`, `Educational Organization`, `Organization Member`, `Organization Member Role`, `Organization Capability`, `Act of Employment by an Organization` (11 candidates) |
| `facility` | `Facility` plus 30 compounds (`Product Transport Facility`, `Healthcare Facility`, `Military Headquarters Facility`, …) |
| `occupation` | `Occupation Role` (single candidate → resolved at 1.0) |
| `event` | `Event Status Nominal Information Content Entity` (single candidate → resolved at 1.0) |
| `animal` | `Animal` (single candidate → resolved at 1.0) |
| `product` | `Product Transport Facility` (single candidate → resolved at 1.0) |

Three of the seven — `occupation`, `event`, `product` — had exactly *one*
fragment candidate each, so `_resolve_candidate` accepted it outright with no
check at all. That is the reported symptom: `event` never anchors to the CCO
class that means "an event" (there is no such single class — CCO models it as
`Act`), it anchors to an information-content-entity subclass that happens to
contain the word "event" among five others. `person` and `organization` had
several candidates and would at least reach ranking, but ranking is over
label-fragment *definitions*, not over the real class.

## After the fix: CCO, curated coarse aliases

The overlay's `coarseAliases` block wins outright, no ranking:

| Coarse type | Overlay alias | Resolves in CCO to |
| --- | --- | --- |
| `person` | `Person` | `Person` (`ont00001262`) |
| `organization` | `Organization` | `Organization` (`ont00001180`) |
| `facility` | `Facility` | `Facility` (`ont00000192`) |
| `occupation` | `Occupation Role` | `Occupation Role` (`ont00000984`) |
| `event` | `Act` | `Act` (`ont00000005`) |
| `animal` | `Animal` | `Animal` (`ont00000562`) |
| `product` | `Material Artifact` | `Material Artifact` (`ont00000995`) |
| `entity` | `""` (empty) | **deliberate abstention** — `entity` is not a typeable coarse concept, and the empty string is kept rather than filtered so the abstention is intentional, not a miss. |

All seven anchor to the CCO class that actually means the coarse concept
(`entity` deliberately abstaining), at score 1.0, with no accidental
competitor in the running. `product`'s alias was corrected in the same diff
(`conversational.json`: `"Artifact"` -> `"Material Artifact"`, the only CCO
class that actually carries that name — `Artifact` alone does not exist in
the bundled digest, only compounds), so it is not a residual gap.

## Under the personal profile

`GRAPHKNOWS_ONTOLOGY` pointed at `personal/personal-profile.json` still loads
the **same bundled overlay** (`overlay_source` is a separate setting), so the
same CCO-shaped `coarseAliases` block is consulted first — but when its
target label names no class in the loaded ontology, `_lexical_candidates`
falls through to the normal exact-label / altLabel build instead of
abstaining, so the profile's own classes still get a chance:

| Coarse type | Overlay alias | Resolves in personal profile to |
| --- | --- | --- |
| `person` | `Person` | `Person` (`schema.org/Person`) — curated alias hits directly |
| `organization` | `Organization` | `Organization` (`schema.org/Organization`) — curated alias hits directly |
| `facility` | `Facility` | **unresolved** — no `Facility` class in the 9-class profile, alias falls through, no exact-label match either |
| `occupation` | `Occupation Role` | `Occupation` (`schema.org/Occupation`) — alias doesn't resolve (profile has no `Occupation Role`), falls through to the exact-label match |
| `event` | `Act` | `Event` (`schema.org/Event`) — same fallthrough |
| `animal` | `Animal` | **unresolved** — no `Animal` class in the profile, alias falls through, no exact-label match either |
| `product` | `Material Artifact` | `Product` (`schema.org/Product`) — same fallthrough |

5 of 7 coarse types anchor under the personal profile — the two remaining
gaps (`facility`, `animal`) are a genuine vocabulary gap in the 9-class
profile, not a code defect: those classes simply don't exist there under any
name. The curated alias is still authored in CCO's vocabulary and is still
checked first, but no longer short-circuits a non-CCO ontology's own
exact-label typing when the alias itself doesn't resolve.

## The finding

CCO's failure at entity typing was never that it lacks the classes. It has
`Act`, `Material Artifact`, `Occupation Role`, `Person`, `Facility` — a class
for every one of the extractor's guaranteed coarse types. What was missing
was a bridge from the extractor's plain English words to CCO's own naming,
and in that gap a label fragment did the bridging by accident, anchoring a
generic type onto whichever specific class happened to contain the word.

## Decision

**CCO stays the entity-typing default** (`GRAPHKNOWS_ONTOLOGY`). This is a
knob independent of the relation vocabulary (`GRAPHKNOWS_RELATION_ONTOLOGY`,
moved to the personal profile by #137) — the two questions ("what type is
this entity" vs. "what relation connects these two entities") are answered by
different vocabularies for a reason documented in
`evaluation/audit/baseline/relation-vocabulary-ab.md`, and nothing in this
census changes that reasoning. The personal profile remains the alternative
for anyone who wants a 9-class flat typing vocabulary — with the caveat above:
choosing it for entity typing gets 5 of 7 coarse types for free (`person`,
`organization` from the curated alias directly; `occupation`, `event`,
`product` via fallthrough to the profile's own exact-label match);
`facility` and `animal` still need either their own curated aliases or
classes added to the profile.

**The residual weakness, named honestly:** CCO puts `Person` under `Animal`
(`Person -[:broader]-> Animal -[:broader]-> Organism -> object -> material
entity -> independent continuant -> continuant -> entity`) — correct in CCO's
upper-ontology terms (a person *is* biologically an animal), and useless for
personal memory, where nobody wants "who are the animals in this
conversation" to include their sister. This is not a reason to swap
vocabularies: CCO's `broader` links are already how the graph generalises
(`GraphStore._narrower_scores` walks exactly one hop down them at query time),
so the seam for fixing this is authoring a better `broader` target for
`Person` in the overlay — the same mechanism `coarseAliases` already uses to
correct a mapping — not a wholesale vocabulary swap.
