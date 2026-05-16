# Wiki Schema — Graphknows

**Domain:** Deep research agent for political intelligence — exhaustive topic research → per-session KG + vector store → structured briefing.

---

## Directory Structure

```
llm-wiki/
├── SCHEMA.md           # This file — conventions and structure
├── index.md            # Catalog — every page with one-line summary
├── log.md              # Append-only action log
│
├── architecture/       # Current authoritative design docs (living, updated as system evolves)
├── adr/                # Architecture Decision Records — immutable, numbered, dated
├── product/            # Vision, roadmap, business, UX design
├── research/           # Domain knowledge, data sources, evaluation, methodology
├── ops/                # Deployment, admin, technical debt, infra notes
├── entities/           # Named entities: tools, models, services, organisations
└── raw/                # Immutable source material — never modify
    └── papers/
```

---

## Folder Conventions

| Folder | Contents | Mutable? |
|---|---|---|
| `architecture/` | Service design, data flows, API contracts, component interfaces |  Update when system changes |
| `adr/` | Decision records: numbered `NNN-slug.md`, record *why* not *what* |  Immutable after merge |
| `product/` | Roadmap, business plan, market, frontend/UX design |  Update as priorities shift |
| `research/` | EU sources, KG methodology, evaluation results, thesis findings |  Update after experiments |
| `ops/` | Infra setup, admin tasks, technical debt register |  Update as work progresses |
| `entities/` | Tools, models, external services — stable reference |  Update when understanding deepens |
| `raw/` | Downloaded papers, scraped content — never edit |  Immutable |

---

## File Naming

- Lowercase, hyphens, no spaces
- `architecture/` and `product/` and `research/`: descriptive slug, no dates
- `adr/`: `NNN-slug.md` (e.g. `001-langgraph-agent.md`) — sequential, never renumbered
- `ops/` and `entities/`: descriptive slug
- No dated session notes in the wiki — those belong in `log.md`

---

## Frontmatter (required on every page)

```yaml
---
title: Human-readable title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: architecture | adr | product | research | ops | entity
tags: [from taxonomy below]
confidence: high | medium | low
---
```

ADRs add:
```yaml
status: accepted | superseded | deprecated
superseded_by: NNN-slug   # if applicable
```

---

## Tag Taxonomy

**Domain:** `agent`, `langgraph`, `graphgen`, `graphrag`, `retrieval`, `graph`, `nlp`, `embedding`
**Infrastructure:** `neo4j`, `memgraph`, `pgvector`, `docker`, `fastapi`
**Product:** `frontend`, `ux`, `roadmap`, `business`, `market`
**Process:** `architecture`, `planning`, `evaluation`, `thesis`, `search`, `ingestion`
**Data:** `eu`, `sources`, `taxonomy`, `corpus`
**Ops:** `admin`, `deployment`, `debt`, `testing`

---

## When to Write What

| Situation | Where to write |
|---|---|
| System design changes | Update `architecture/` page |
| New technical decision made | Create `adr/NNN-slug.md` |
| Priority or roadmap shifts | Update `product/roadmap.md` |
| Experiment / eval result | Update `research/` page |
| Bug found, infra changed | Append to `ops/technical-debt.md` or `ops/administration.md` |
| Session actions summary | Append to `log.md` only |
| Downloaded paper | Save to `raw/papers/`, extract insights to `research/` |

**Never create a dated session note as a wiki page.** If it happened and is done, it goes in `log.md`. If the outcome should persist, distill it into the appropriate folder.

---

## Log Format

Append-only. Each entry:

```
## [YYYY-MM-DD] <verb> | <short description>
- bullet summary of what happened / was decided
```

Verbs: `create`, `update`, `pivot`, `impl`, `fix`, `eval`, `research`, `delete`
