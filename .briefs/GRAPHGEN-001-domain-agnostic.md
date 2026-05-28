# GRAPHGEN-001 — Make graphgen Domain-Agnostic (Phase C)

## Context
Working directory: `/root/graphknows`. Confirm this before doing anything.

The graphgen batch corpus pipeline is a powerful generic NER+RE+graph building engine.
Its only domain-specific parts are:
1. The system prompt in `graphknows/graphgen/pipeline/summarization/core.py` — references
   "intelligence analyst specializing in graph-based pattern detection" (EU flavored)
2. The default taxonomy — `graphknows/graphgen/config/taxonomy.py::get_default_taxonomy()`
   hardcodes `taxonomy.eu-politics.yaml` as the default

The goal: make graphgen fully domain-agnostic by default, while keeping EU as an optional example.

## Task 1: Replace EU system prompt in summarization

**File:** `graphknows/graphgen/pipeline/summarization/core.py`

Find the system prompt (likely a string like "You are an intelligence analyst...").
Replace it with a neutral knowledge synthesis prompt:

```
You are a knowledge synthesis assistant. Your task is to analyze entities, relationships,
and patterns within the knowledge graph community you have been given and produce a concise,
factual summary of the key themes, entities, and connections. Do not assume any particular
domain. Write clearly for a general technical audience.
```

**Important:** Only change the system prompt string. Do not modify any other logic,
structure, or parameters in this file.

## Task 2: Add taxonomy.generic.yaml

**Create file:** `graphknows/graphgen/config/taxonomy.generic.yaml`

Model it after the structure of `graphknows/graphgen/config/taxonomy.eu-politics.yaml`
but with fully generic entity types and relations suitable for any domain:

Entity types to include:
- PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, DATE, QUANTITY

Relation types to include (representative set):
- PART_OF, RELATED_TO, CAUSES, LOCATED_IN, AFFILIATED_WITH, PRECEDED_BY, SUCCEEDED_BY,
  PRODUCED_BY, DESCRIBED_BY, CONTRASTS_WITH

Read the eu-politics.yaml first to understand the exact YAML structure, then follow it.

## Task 3: Update get_default_taxonomy()

**File:** `graphknows/graphgen/config/taxonomy.py`

Update `get_default_taxonomy()` to:
1. Check env var `GRAPHKNOWS_TAXONOMY_FILE` first (already exists — keep this)
2. If not set, default to `taxonomy.generic.yaml` (change from `taxonomy.eu-politics.yaml`)
3. Keep `taxonomy.eu-politics.yaml` as a bundled domain example (do not delete it)

The eu-politics taxonomy should still be selectable via:
```bash
GRAPHKNOWS_TAXONOMY_FILE=taxonomy.eu-politics.yaml graphknows-cli graphgen ...
```

## What NOT to touch
- The pipeline machinery: NER, RE, graph cleaning, community detection, embeddings, analytics
- `iterative_orchestrator.py`
- `taxonomy.eu-politics.yaml` — keep it as a domain example
- Any files outside `graphknows/graphgen/`
- Any test files (there are unlikely to be tests for the prompt string — do not add tests
  for the prompt content itself, only for `get_default_taxonomy()` path resolution if it's
  not already tested)

## Acceptance Criteria
1. `graphknows/graphgen/pipeline/summarization/core.py` system prompt is neutral, not EU-specific.
2. `graphknows/graphgen/config/taxonomy.generic.yaml` exists with generic entity/relation types.
3. `get_default_taxonomy()` returns the generic taxonomy by default (when env var not set).
4. `GRAPHKNOWS_TAXONOMY_FILE=taxonomy.eu-politics.yaml` still works (EU taxonomy still selectable).
5. `uv run pytest tests/ -x -q -m "not integration"` passes — no new failures.
6. `ruff format graphknows/ && ruff check --no-fix graphknows/` — zero new errors.

## Coding Standards
Read before starting:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`

## Scope
Only modify the files listed. Do not refactor, rename, or improve anything else.

## End-of-task summary required
```
Files changed: <list with one-line description>
Files created: <list>
Files NOT touched: <confirm>
Follow-up needed: <list or "none">
```
