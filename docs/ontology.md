# Ontology Injection

Ontology injection is what keeps the `llm_assisted` knowledge graph
domain-specific. Instead of trusting an LLM to invent relation names, graphknows
lets the LLM extract free-form triplets and then **maps every predicate to the
closest property in a controlled vocabulary** loaded from an RDF/OWL ontology you
supply. Triplets whose predicate cannot be mapped above a similarity threshold are
dropped. The result is a graph whose edge types come from *your* ontology, not
from whatever phrasing the model happened to produce.

This is only active in `llm_assisted` mode. In `llm_free` mode there is no
ontology; relations come from a spaCy extractor filtered against a flat taxonomy.

## Extract-then-map

The core idea is a two-step pipeline per chunk:

1. **Extract.** A DSPy module prompts the LLM to emit `(source, relation, target)`
   triplets over the chunk text. The LLM is free to phrase the predicate however
   it likes — `"is friends with"`, `"works at"`, `"is the mother of"`.
2. **Map.** Each free-form predicate is matched to the closest controlled-vocabulary
   property from the loaded ontology. If the best match clears the similarity
   threshold, the predicate is rewritten to the canonical label. If not, the whole
   triplet is discarded.

This keeps extraction recall high (the LLM is not fighting a rigid grammar) while
constraining the written graph to a fixed, domain-specific set of edge types.

## The pieces

All ontology code lives under `graphknows/extraction/ontology/`.

### Loader — `loader.py`

`load_rdf_ontology_file(path)` parses one explicit RDF/OWL/Turtle file with
`rdflib` and returns an `RdfOntologyManifest` with:

- `class_labels` — labels of `owl:Class` / `rdfs:Class` subjects,
- `property_labels` — labels of `owl:ObjectProperty`, `owl:DatatypeProperty`, and
  `rdf:Property` subjects,
- `context` — a compact JSON string of the file, classes, and properties for the
  LLM prompt.

Labels come from `rdfs:label` where present, otherwise the fragment after `#` or
the last path segment of the URI. The loader is intentionally **strict** for
LLM-assisted extraction: a missing file, or a file with no class or property
labels, is an error rather than a silent fallback to defaults.

(There is also an `OntologyLoader` / `OntologyManifest` pair for the bundled
registry of ontologies, used by the router and by unit tests.)

### Predicate mapper + filter — `relation_filter.py`

`OntologyPredicateMapper` does the mapping. `map_predicate(predicate)` returns
`(canonical_label, score)` via a three-stage, fully-local pipeline (no network
calls):

1. **Exact match** on the normalised (lowercased, punctuation-stripped) label →
   score `1.0`.
2. **Surface-alias lookup** — common natural-language phrases (e.g. `"belongs to"`
   → `member_of`), restricted to aliases whose target is actually in this
   ontology → score `0.95`.
3. **Character-n-gram TF-IDF cosine similarity** against every ontology property
   label; the best label is returned if its score meets `min_similarity`,
   otherwise `(None, score)` and the triplet is dropped.

`RelationOntologyFilter` is the simpler companion: given a list of allowed
property labels, `is_allowed(label)` / `filter(...)` keep only relations whose
label is in the allow-list (an empty list makes it a pass-through). This is the
filter the `llm_free` STM path uses against the taxonomy vocabulary.

### Router (optional) — `router.py`

`OntologyRouter.select(text)` picks the best ontology from a *collection* for a
given document, combining a keyword-frequency heuristic with embedding cosine
similarity (`alpha` weights the two; embeddings degrade gracefully to
keyword-only if no embedder is available). `select_rdf_file(text)` returns the
chosen ontology's primary RDF file path, and `select_multi(text, top_k)` returns
several for multi-domain documents. Use this when you have many ontologies and
want per-document routing rather than one fixed file.

### DSPy wiring — `graphknows/extraction/relations/dspy.py`

`DSPyExtractor` ties it together. On construction, `_load_ontology` reads
`config["ontology"]["file"]`:

- **File configured** → `load_rdf_ontology_file` populates `_ontology_classes`
  and `_ontology_relation_types`, and an `OntologyPredicateMapper` is built from
  the raw property labels (so surface forms like `"is friends with"` can match
  `friend_of`). A SKOS-enriched context string is built for the prompt.
- **No file** → the mapper is built from an open-domain relation-type list with a
  slightly higher threshold (`min_similarity=0.35`), so extract-then-map still
  applies, just with a broad vocabulary.

During extraction, `_apply_extract_then_map(raw)` calls
`mapper.map_predicate(...)`: on success the triplet's `relation` is replaced with
the canonical label (and `_mapped_from` / `_map_score` recorded when it was
remapped); on failure the triplet is dropped with reason
`predicate_unmappable(score=...)`.

## Turning it on

```bash
pip install "graphknows[assisted]"     # pulls in dspy + rdflib
```

Then set the environment (or `.env`):

```bash
GRAPHKNOWS_MODE=llm_assisted
GRAPHKNOWS_LLM_MODEL=openai/gpt-4o-mini      # any litellm model string
GRAPHKNOWS_LLM_API_KEY=sk-...
GRAPHKNOWS_ONTOLOGY_FILE=/path/to/ontology.ttl
```

`GRAPHKNOWS_ONTOLOGY_FILE` is read into `config["ontology"]["file"]` and feeds the
loader. Without it, `llm_assisted` still runs but maps against the open-domain
vocabulary instead of your domain ontology. (In production, `llm_assisted`
requires `GRAPHKNOWS_LLM_API_KEY` — `check_production_secrets` raises otherwise.)

## Worked example

A minimal personal-relationships ontology in Turtle (`personal.ttl`):

```turtle
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/personal#> .

ex:Person      a owl:Class ; rdfs:label "Person" .
ex:Company     a owl:Class ; rdfs:label "Company" .
ex:Place       a owl:Class ; rdfs:label "Place" .

ex:friend_of   a owl:ObjectProperty ; rdfs:label "friend_of" .
ex:employed_by a owl:ObjectProperty ; rdfs:label "employed_by" .
ex:lives_in    a owl:ObjectProperty ; rdfs:label "lives_in" .
```

`load_rdf_ontology_file("personal.ttl")` yields:

- `class_labels = ["Company", "Person", "Place"]`
- `property_labels = ["employed_by", "friend_of", "lives_in"]`

Now suppose the LLM extracts this triplet from a chunk:

```
("Caroline", "is friends with", "Melanie")
```

`OntologyPredicateMapper.map_predicate("is friends with")` runs:

1. Exact match against `{employed_by, friend_of, lives_in}` — no hit.
2. Surface-alias lookup — `"is friends with"` / `"friend of"` resolves to the
   `friend_of` label present in this ontology → canonical `friend_of`, score `0.95`.
   (Had no alias applied, stage 3's character-n-gram TF-IDF cosine between
   `isfriendswith` and `friendof` would still score above the threshold on the
   shared `friend` n-grams.)

The triplet is rewritten to `("Caroline", "friend_of", "Melanie")` and written to
the graph as a bi-temporal `friend_of` edge between the two `ENTITY` nodes.

A triplet like `("Caroline", "vibed with", "the sunset")` has no exact match, no
alias, and no property n-gram-similar enough to clear the threshold — so
`map_predicate` returns `(None, low_score)` and the triplet is **dropped**,
keeping the graph clean.
