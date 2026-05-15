# ADR-003: Ontology-Guided KG Extraction

**Status**: Accepted  
**Date**: 2026-05-13  
**Replaces**: t_b2627d7f (LKIF-only typed relations — too narrow)

---

## Context

The original t_b2627d7f proposed LKIF-only typed relation edges for legal/policy documents.
The user wants the extraction to be **domain-agnostic** — governed by pre-downloaded ontology
files so the same pipeline works for legal, finance, healthcare, science, politics, etc.

Research findings (Perplexity, 2026-05-13):
- **OntoGPT** (Monarch Initiative) is the closest existing framework — LLM + ontology grounding
- **SPIRES** pattern: LLM extraction constrained by OWL class hierarchy
- Best production approach: **GLiNER entity types from ontology labels** + **DSPy relation extraction with ontology schema as context**
- Ontology routing: embedding similarity (fast) → keyword heuristics → LLM classifier (fallback)

---

## Decision

### Ontology Registry

Pre-bundle a curated set of ontologies under `services/agent/ontology/assets/`:

```
services/agent/ontology/assets/
  registry.json           # metadata + pre-computed embeddings for routing
  legal/
    lkif-core.ttl         # LKIF (legal framework, contracts, jurisdiction)
    lkif-summary.jsonld   # compressed for LLM context window
  politics/
    prov-o.ttl            # provenance, actors, affiliations
    political.jsonld      # custom: parties, roles, offices, votes
  finance/
    fibo-entities.ttl     # FIBO financial instruments & organizations
    fibo-summary.jsonld
  science/
    schema-org-core.jsonld # Schema.org creative works, events, organizations
  general/
    wikidata-core.ttl     # cross-domain fallback (top 2000 classes)
```

**Format**: Turtle (`.ttl`) for full OWL; JSON-LD summary (500–2000 classes) for LLM prompts.

### Routing

```
OntologyRouter.select(doc_text: str) -> OntologyManifest
  1. Embedding similarity: doc_embedding vs. per-ontology description embeddings
  2. Keyword heuristics: domain keyword frequency (fast, cheap)
  3. LLM classifier (dspy.ChainOfThought): only if top sim < 0.7
  Returns top-1 or top-2 ontologies with weights
```

### Extraction Pipeline

```
Document text
  → OntologyRouter.select()          → OntologyManifest (class labels + relation types)
  → GLiNER(entity_types=classes)     → Entity list with typed labels
  → DSPy OntologyRelExtractor(
        text, entities,
        ontology_schema=jsonld_summary
    )                                  → Relation triples (subj, pred, obj, confidence)
  → Neo4j writer: MERGE on (name, label), CREATE typed REL
```

### DSPy Module

```python
class OntologyRelExtractor(dspy.Module):
    """Extract typed relations guided by an ontology schema."""
    
    def __init__(self) -> None:
        self.extract = dspy.ChainOfThought(
            "text, entities, ontology_schema -> relations"
        )
    
    def forward(self, text: str, entities: list[dict], ontology_schema: str) -> dspy.Prediction:
        return self.extract(
            text=text,
            entities=json.dumps(entities),
            ontology_schema=ontology_schema,
        )
```

Output relations: `[{"subject": str, "predicate": str, "object": str, "confidence": float}]`

### Neo4j Writer

```cypher
MERGE (s:Entity {name: $subj_name, label: $subj_label})
MERGE (o:Entity {name: $obj_name, label: $obj_label})
MERGE (s)-[r:ONTOLOGY_REL {type: $predicate, ontology: $ontology_id}]->(o)
ON CREATE SET r.confidence = $confidence, r.session_id = $session_id
```

---

## Implementation Plan

### Phase 1 — Ontology registry + loader (t_b2627d7f renamed)
- `services/agent/ontology/assets/` directory with `registry.json`
- Download 5 base ontologies (LKIF, PROV-O, FIBO-entities, Schema.org, Wikidata-core)
- `OntologyLoader`: parses Turtle with `rdflib`, extracts class labels and property labels
- Unit tests: loader parses each bundled ontology, returns non-empty class list

### Phase 2 — Router
- `OntologyRouter`: embedding sim (BAAI/bge-small) + keyword heuristics
- Pre-compute description embeddings at startup, cache in `registry.json`
- Unit tests: legal doc → lkif, finance doc → fibo, medical doc → snomed-lite

### Phase 3 — DSPy extractor
- `OntologyRelExtractor(dspy.Module)` with Groq backend
- Integrate into `investigate_node`: after GLiNER entity extraction, call DSPy extractor
- Write relations to Neo4j with `ontology_id` property
- Unit tests: mock DSPy, assert triples written to Neo4j

### Phase 4 — Wire to investigate_node
- `make_investigate_node` gains `ontology_router` and `rel_extractor` params
- Emit `relations_extracted` WSEvent with count + ontology used
- Frontend shows ontology badge in graph legend

---

## Ontology Sources

| Ontology | Source URL | License |
|---|---|---|
| LKIF Core | https://raw.githubusercontent.com/RinkeHoekstra/lkif-core/master/lkif-core.owl | LGPL |
| PROV-O | https://www.w3.org/ns/prov.ttl | W3C |
| FIBO Entities | https://spec.edmcouncil.org/fibo/ontology/FND/Parties/Parties.rdf | CC BY 4.0 |
| Schema.org | https://schema.org/version/latest/schemaorg-current-https.jsonld | CC BY-SA 4.0 |
| Wikidata core | via wikidata SPARQL (top 2000 P* and Q classes) | CC0 |

---

## Rejected Alternatives

- **LKIF-only (original t_b2627d7f)**: too narrow, no domain generalisation
- **SPARQL triple store (Jena/Virtuoso)**: adds infra complexity not justified at this scale
- **Fine-tuning GLiNER per ontology**: requires labelled data we don't have
