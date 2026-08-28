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
`load_ontology_terms()` reads both, along with `rdfs:domain`, `rdfs:range` and
`rdfs:subClassOf`, into `OntologyTerm` records. `OntologyIndex` embeds each term
once as `"Label. definition"` and caches the matrix to disk, keyed by embed model
+ dimension + a content hash of the sources.

That gives one cheap operation: *given an embedding, which ontology terms is this
about?* It is the same device the FrameNet layer uses, and it reuses the chunk
embedding already computed at ingest, so injection adds no embedding call on the
chunk path.

## What happens at ingest

Per chunk:

1. **Select.** Top-k classes and properties by cosine against the chunk's
   embedding. Gated by top-k alone, not an absolute threshold — matching a
   concrete utterance to an abstract class definition is a weak signal (top-1
   cosine median ≈ 0.20 on conversational text with bge-small), and an
   over-broad candidate list costs nothing.
2. **Inject.** Those labels are **unioned into** the spaCy-mined label space
   handed to the extractor. Never substituted: the mined vocabulary keeps
   precedence, so injection can only add candidates.
3. **Map.** Each extracted entity type is mapped to an ontology class, and each
   relation predicate to an ontology property.
4. **Link.** `ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`, carrying the grounding's
   full provenance (`channel`, `version`, `matcher`, `score`, `rank`,
   `chunk_id`, `span`); `CHUNK-[:MATCHES_CLASS {sim}]->ONTOLOGY_CLASS`; and
   `REL.ontology_property` / `REL.map_score`.

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
  from embedding similarity against the whole class vocabulary. A single
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

Edges: `BROADER` (class → superclass — the SKOS-shaped subsumption taxonomy,
walked one hop DOWN at a discount by the query-side subclass expansion),
`INSTANCE_OF` (entity → class — `score`, plus the grounding provenance
`channel`, `version`, `matcher`, `rank`, `chunk_id`, `span`), `MATCHES_CLASS`
(chunk → class, `.sim`).

Written once per namespace at schema-ensure time, count-guarded so a restart is
a no-op. The DDL is unconditional and additive, so an existing namespace gains
the types (empty) on its next `ensure_schema`.

**Changing `GRAPHKNOWS_ONTOLOGY` on an existing namespace accumulates rather
than replaces.** Each ontology is keyed by a content digest, so pointing a
namespace at a second one adds its terms alongside the first (one `ONTOLOGY`
node each). This is deliberate: chunks already tagged under the old ontology
keep valid `MATCHES_CLASS` edges, and distinct ontologies have distinct URIs.
But it does mean `OntologyChannel` will vector-search *both* vocabularies. Drop
and re-ingest the namespace if you want a clean switch.

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
