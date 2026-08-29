# Ontology Injection

GraphKnows always runs with an ontology: it ships CCO as the default (see
[The bundled ontology](#the-bundled-ontology)). Point `GRAPHKNOWS_ONTOLOGY` at
your own RDF/OWL file **or a folder of them** to replace it, and GraphKnows will
steer extraction toward your domain's vocabulary, link the resulting entities and
relations to the ontology terms they instantiate, and persist the ontology itself
as part of the graph.

This is **mode-independent**: it works identically in `llm_free` and
`llm_assisted`. The two modes differ in what runs *on top* (see
[modes.md](modes.md)), not in whether an ontology applies.

```bash
GRAPHKNOWS_ONTOLOGY=/path/to/ontology.ttl     # or /path/to/ontologies/
```

`GRAPHKNOWS_ONTOLOGY_FILE` is a deprecated alias accepting a single file.

## The mechanism: definitions, embedded

An ontology's load-bearing content is not its labels but its **definitions** —
the `skos:definition` / `rdfs:comment` text describing each class and property.
`load_ontology_terms()` reads both, along with `rdfs:domain`, `rdfs:range`,
`rdfs:subClassOf`, `skos:altLabel` and `skos:example`, into `OntologyTerm`
records. `OntologyIndex` embeds each term once as its label, definition,
alt labels and examples joined together, and caches the matrix to disk, keyed
by embed model + dimension + a content hash of the sources.

That gives one cheap operation: *given an embedding, which ontology terms is this
about?* It is the same device the FrameNet layer uses, and it reuses the chunk
embedding already computed at ingest, so injection adds no embedding call on the
chunk path.

Sharing the device is not sharing meaning: the ontology and FrameNet each produce
a separate, unjoined signal from it, and nothing in the graph maps a FrameNet
frame or frame element onto an ontology class.

## What happens at ingest

Per chunk:

1. **Select.** Top-k classes and properties by cosine against the chunk's
   embedding, gated by a similarity floor (`ontology_hint_min_sim`) rather than
   top-k alone — needed once the class space grew from a handful of bundled
   classes to CCO's 1437, where an unfloored top-12 hands twelve arbitrary
   labels to a chunk that says "Thanks, Gina!". These are candidate labels
   offered to the extractor, not a persisted claim; the stricter, separately
   anchored decision that IS persisted is `MATCHES_CLASS` in step 4.
2. **Inject.** Those labels are **unioned into** the spaCy-mined label space
   handed to the extractor. Never substituted: the mined vocabulary keeps
   precedence, so injection can only add candidates.
3. **Map.** Each extracted entity type is mapped to an ontology class, and each
   relation predicate to an ontology property.
4. **Link.** `ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`, carrying the grounding's
   full provenance (`channel`, `version`, `matcher`, `score`, `rank`,
   `chunk_id`, `span`); `CHUNK-[:MATCHES_CLASS {sim}]->ONTOLOGY_CLASS`; and
   `REL.ontology_property` / `REL.map_score`. `version` is the loaded
   ontology's content digest (`OntologyIndex.digest`), not its file path, so
   replacing a file in place still produces a distinguishable version.

   `MATCHES_CLASS` obeys the anchor law, not plain cosine: a class is tagged
   onto a chunk only when the chunk's own text lexically EVOKES it
   (`lexical.evoked`, the same mechanism `INSTANCE_OF` typing anchors on), and
   cosine is then used only to RANK the classes lexis already admitted, over
   the ontology's whole vocabulary rather than an unanchored top-k (the
   anchored class is almost never in the top-k of a whole-vocabulary cosine
   lookup). This matters because absolute cosine over CCO's 1437 class
   definitions is uncalibrated — 20 real conv-30 turns put the whole class
   space at median 0.409 / p95 0.493 — and unanchored, every accepted tag
   landed in a near-uniform 0.557-0.590 band where a BFO upper class like
   `Curvilinear Motion` could outscore `Dance Studio` on a chunk that shares no
   vocabulary with it at all. A chunk that evokes no class is tagged with
   nothing rather than the top cosine hit, and counted as the
   `unanchored_class_tag` abstention; the accepted sims for the run are logged
   as one INFO line (count, min, max, spread) so a degenerate band — spread
   under ~0.05 — is visible without re-running the bug.

### Enrichment, not filtering

An entity or relation that maps to nothing is **still written**, under its
extracted label. This is the critical property: a controlled vocabulary that
*replaced* the open one would make everything outside your ontology invisible.
Measured on a sample conversation, ontology-on produced 26 entities against 20
without, with `nurse`/`Bruno`/`cilantro` correctly left unmapped rather than
forced into a wrong class.

The two mappers are deliberately different:

- **Predicates** use `OntologyPredicateMapper` — exact match (1.0), then a
  surface-alias table (`"is friends with"` → `friend_of`, 0.95), then
  character-n-gram TF-IDF cosine. Suits multi-word phrases.
- **Entity types** are grounded in two stages, memoised per distinct type.
  The candidate set comes only from lexical evidence — an exact label match,
  or a SKOS altLabel once the corpus-frequency lexicon has warmed up — never
  from embedding similarity against the whole class vocabulary. An altLabel
  candidate only counts when every content word of its class label is also a
  content word of the extracted type, so a label that merely *contains* the
  type as one word among several ("product" inside "Product Transport
  Facility") cannot anchor on its own. The bundled overlay's `coarseAliases`
  block additionally maps the extractor's seven guaranteed coarse types
  (`person`, `organization`, `facility`, `occupation`, `event`, `animal`,
  `product`) to the CCO class each one means — `event` to `Act`, `product` to
  `Material Artifact` — ahead of that lexical build, since three of the seven
  are not CCO class labels verbatim; a coarse type mapped to `""` (`entity`)
  abstains outright rather than fall back to a fragment match. A single
  candidate is the answer (`matcher="lexical"`, score 1.0). Several
  candidates are ranked by embedding, but only against each other's
  definitions, and accepted only if the top one clears a similarity floor and
  beats the runner-up by a margin (`matcher="lexical+cosine"`). A type with no
  lexical candidate at all is left unmapped rather than cosine-matched against
  the full vocabulary: N-gram similarity is actively wrong on single-word type
  labels (it confidently returns `LOCATION → Organization` and
  `ANIMAL → Goal`), and an unanchored embedding match is no more trustworthy —
  so an unrecognised type stays unmapped, which is the right answer for a type
  the ontology has no class for.

## What happens at retrieval

The ontology is queryable because it is *in* the graph. `OntologyChannel`
vector-searches `ONTOLOGY_CLASS.embedding` with the query embedding and
traverses `MATCHES_CLASS` to the chunks tagged with the matching classes,
contributing candidates to the RRF fusion. So "where did they go" can reach
chunks tagged `Place` without the word "place" occurring anywhere.

The channel is registered only when an ontology is configured, and it drops
classes that tag more than half a session's chunks — in a personal corpus almost
every chunk is "about" a `Person`, and an unguarded `Person` class returns the
entire conversation.

## Schema

| Type | Key fields |
|---|---|
| `ONTOLOGY` | `id` (content digest), `source`, `term_count` |
| `ONTOLOGY_CLASS` | `uri`, `label`, `definition`, `embedding` (LSM_VECTOR COSINE) |
| `ONTOLOGY_PROPERTY` | `uri`, `label`, `definition`, `domain`, `range`, `embedding` |
| `SOURCE` | `id` (`"<channel>@<version>"`), `channel`, `version`, `source_hash`, `imported_at`, `imported`, `dropped_expressions`, `dropped_fe_mappings` |

Edges: `BROADER` (class → superclass — the SKOS-shaped subsumption taxonomy,
walked one hop DOWN at a discount by the query-side subclass expansion),
`INSTANCE_OF` (entity → class — `score`, plus the grounding provenance
`channel`, `version`, `matcher`, `rank`, `chunk_id`, `span`, and `superseded`
once a rebind promotes a newer version), `MATCHES_CLASS` (chunk → class,
`.sim`).

Written once per namespace at schema-ensure time, count-guarded so a restart is
a no-op. The DDL is unconditional and additive, so an existing namespace gains
the types (empty) on its next `ensure_schema`.

**Why all 1437 CCO classes get written, not only the ones a chunk actually
matched.** `GraphStore.write_ontology` writes every class vertex, its
definition embedding and its `BROADER` edges up front, whether or not any
chunk in the namespace ever gets tagged with it. That is not wasted
materialisation: a class no chunk ever matched directly is the entry point
that makes its grounded descendants reachable from a general query.
`GraphStore._narrower_scores` walks one hop DOWN the `BROADER` edges at a
discount, so a query that matches `Facility` reaches a chunk tagged only
`Dance Studio`, even though `Facility` itself tags nothing in that namespace —
the taxonomy has to already be there for that hop to exist. Two honest limits:
that reader sits behind `GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL`, default-off
pending its A/B (see [modes.md](modes.md) / [configuration.md](configuration.md)),
and a class with no grounded descendant in a given namespace genuinely
contributes nothing there — an ungrounded class is dead weight only for the
namespaces that never populate its subtree, not for the mechanism in general.

**Every ontology import mints a `SOURCE` vertex** recording the (channel,
version) it produced, keyed so a re-import under a changed file MERGEs a new
row rather than overwriting the one it replaces. `Memory.stats()` reports
these under `"sources"`.

**Changing `GRAPHKNOWS_ONTOLOGY` on an existing namespace accumulates rather
than replaces.** Each ontology is keyed by a content digest, so pointing a
namespace at a second one adds its terms alongside the first (one `ONTOLOGY`
node each). This is deliberate: chunks already tagged under the old ontology
keep valid `MATCHES_CLASS` edges, and distinct ontologies have distinct URIs.
But it does mean `OntologyChannel` will vector-search *both* vocabularies.

Two versions of the same channel's groundings coexist as separate
`INSTANCE_OF` edges rather than one overwriting the other. To re-ground
already-stored mentions against the ontology currently loaded — without
dropping the namespace — call `Memory.rebind_memory(channel="ontology")`: it
re-resolves each mention, writes the new groundings under the current
version, and flags the superseded ones rather than deleting them. Drop and
re-ingest the namespace instead if you want the old groundings gone, not just
flagged.

## Folders and multiple ontologies

A folder loads every `.ttl` / `.owl` / `.rdf` / `.jsonld` beneath it, and all
terms go into **one** matrix — per-chunk selection then draws from whichever
ontology fits, with no routing stage. Terms are de-duplicated by
`(kind, label)`, keeping the entry that carries a definition, so an ontology
sitting beside its JSON-LD digest does not double up.

Unparseable files in a folder are skipped rather than fatal, and an unusable
`GRAPHKNOWS_ONTOLOGY` logs and continues without injection — enrichment must not
fail an ingest.

## The bundled ontology

The bundled vocabulary is the **Common Core Ontologies (CCO)**, and it is the
*always-on default* — not an opt-in. `GRAPHKNOWS_ONTOLOGY` defaults to
`graphknows/ontology/assets/cco/cco.json`, resolved in
[`graphknows/settings.py`](../graphknows/settings.py) (`_BUNDLED_ONTOLOGY`).
Setting the variable **replaces** that default with your own file or folder.

`graphknows/ontology/assets/` ships:

| Path | What it is |
| --- | --- |
| `cco/cco.json` | The merged CCO digest — the default entity-typing ontology, and the example relation vocabulary you switch back to with `GRAPHKNOWS_RELATION_ONTOLOGY=cco`. |
| `cco/conversational.json` | A SKOS overlay adding everyday concepts CCO does not model, with altLabels that make them lexically reachable and domain/range in CCO's own classes. |
| `cco-modules/` | The 11 CCO modules as `.ttl` (Agent, Event, Time, Geospatial, Quality, Artifact, Facility, …). |
| `cco-imports/`, `cco-extensions/` | Import closure and governance-board extensions. |
| `personal/personal-profile.json` | The default relation vocabulary — 9 classes, 25 object properties, every one domain- and range-typed, sized to be offered whole. |

Its classes and properties are named schema.org-style, in camelCase
(`worksFor`, `EducationalOrganization`). `content_tokens` splits a label on
its case boundaries before lowercasing, so `worksFor` contributes `works` as
a lexical surface rather than the unreadable `worksfor` — the same content-word
matching described above then reaches these labels from ordinary text instead
of only from their literal concatenated spelling.

Which classes count as scaffolding rather than a concrete kind — CCO's
`continuant`/`process`, this profile's `Thing` — is derived from the loaded
ontology's own class hierarchy (near a root, subsuming a large share of its
classes), not hand-listed per vocabulary, so a scaffolding-typed property such
as `permits` or `requires` is excluded from domain/range-driven relation
matching automatically, for CCO, the bundled profile, or your own ontology
alike, with no code change.

There is no by-name registry — point `GRAPHKNOWS_ONTOLOGY` at a path, exactly as
you would at your own file:

```bash
GRAPHKNOWS_ONTOLOGY=graphknows/ontology/assets/cco-modules/AgentOntology.ttl
```

Entity typing and the relation vocabulary are two separate knobs:
`GRAPHKNOWS_ONTOLOGY` controls what types entities and always defaults to CCO;
`GRAPHKNOWS_RELATION_ONTOLOGY` controls what relation properties are drawn
from and defaults to the bundled personal profile. The conv-30 personal-vs-cco
A/B (`evaluation/audit/baseline/relation-vocabulary-ab.md`) found CCO's
enterprise/commercial-frame object properties leave 13.5% of REL edges
unanchored versus the personal profile's 2.5% on personal, two-person
conversation — CCO remains loadable as the relation vocabulary too, as a
documented example, with one env var:

```bash
GRAPHKNOWS_RELATION_ONTOLOGY=cco
```

## llm_assisted: extract-then-map

With `GRAPHKNOWS_DSPY_RELATIONS` on, the LLM additionally extracts free-form
triplets which are then mapped to ontology properties. Here mapping **is** a
filter — an unmappable LLM predicate is dropped
(`predicate_unmappable(score=...)`), because the model is free to invent
phrasing and the point is to constrain it. That is a different contract from the
deterministic pass above, where labels are span-grounded and worth keeping
whether or not they map.

With no ontology configured, DSPy maps against a built-in open-domain vocabulary
at a higher threshold (`min_similarity=0.35`).

## Code map

| Concern | Location |
|---|---|
| Term loading (file or folder, with definitions) | `graphknows/ontology/loader.py` |
| Definition embedding index + disk cache | `graphknows/ontology/catalog.py` |
| Bundled assets | `graphknows/ontology/assets/` |
| Predicate mapper | `graphknows/ingestion/extraction/relations/filter.py` |
| Ingest injection + linking | `graphknows/ingestion/stm/ingest.py` |
| Retrieval channel | `graphknows/channels/ontology.py` |
| Persistence | `graphknows/storage/arcadedb/graph_store.py` |
| DSPy wiring | `graphknows/ingestion/extraction/relations/llm_assisted/dspy.py` |
