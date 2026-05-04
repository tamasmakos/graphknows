---
title: Service Design — services/search/
created: 2026-05-04
updated: 2026-05-04
type: concept
tags: [architecture, search, scraper, service-design]
confidence: high
---

# Service Design — `services/search/`

> Evolved from `scrapers/`. The search service is an **agentic document discovery and ingestion service** — it knows how to find, authenticate, rate-limit, parse, and queue public records from supported official sources. It is not a web crawler; it only targets structured official APIs and portals.

---

## Design Philosophy

- **Official sources only**: parliamentary debates, legislation, regulations, court decisions, government reports. No news, no social media.
- **Idempotent ingestion**: each document has a stable canonical ID (e.g. CELEX number, Hansard sitting reference). Re-scraping the same document is a no-op.
- **Async job queue**: scrape jobs are long-running. POST /scrape returns a job_id; GET /jobs/{id} polls progress.
- **Source adapters**: each source is a self-contained adapter class with a standard interface.

---

## Directory Layout

```
services/search/
  main.py              # FastAPI app
  router.py            # route definitions
  job_queue.py         # async job queue (asyncio + Redis or in-memory for dev)
  adapters/
    base.py            # SourceAdapter ABC: search(), fetch(), parse(), canonical_id()
    ep_debates.py      # EU Parliament debates (EurLex + EP Open Data Portal)
    uk_legislation.py  # UK Acts + Statutory Instruments (legislation.gov.uk API)
    uk_hansard.py      # UK Hansard (parliament.uk) — NEW
    bundestag.py       # German Bundestag (DIP API) — NEW (from scrapers/)
    eurlex.py          # EUR-Lex (EU law) — existing, enhanced
    reference_resolver.py  # follows document cross-references (CELEX, SI refs, UN symbols)
  models.py            # Document, SearchResult, IngestJob Pydantic models
  storage.py           # writes raw JSON to data/raw/{source}/ + tracks state.json
  config.py
  Dockerfile
  pyproject.toml
```

---

## HTTP API

```
GET  /sources              — list supported source adapters + status
POST /search               — search for documents across sources
POST /scrape               — queue a scrape job (one or more document IDs)
GET  /jobs/{job_id}        — poll job status
GET  /jobs/{job_id}/result — get scraped documents when complete
GET  /documents/{id}       — get a previously scraped document by canonical ID
```

### POST /search

```json
// Request
{
  "query": "EU migration policy asylum seekers",
  "sources": ["ep_debates", "eurlex"],
  "date_from": "2022-01-01",
  "date_to": "2024-12-31",
  "limit": 20
}

// Response
{
  "results": [
    {
      "id": "CRE-9-2022-06-08-ITM-003",
      "source": "ep_debates",
      "title": "...",
      "date": "2022-06-08",
      "url": "https://...",
      "excerpt": "...",
      "already_ingested": true
    }
  ]
}
```

### POST /scrape

```json
// Request
{ "document_ids": ["CRE-9-2022-06-08-ITM-003"], "source": "ep_debates" }

// Response
{ "job_id": "job_...", "status": "queued", "estimated_seconds": 12 }
```

---

## Source Adapter Interface

```python
class SourceAdapter(ABC):
    source_id: str                    # "ep_debates" | "uk_legislation" | etc.
    rate_limit: float                 # requests per second
    
    async def search(self, query: SearchQuery) -> list[SearchResult]: ...
    async def fetch(self, document_id: str) -> RawDocument: ...
    def parse(self, raw: RawDocument) -> Document: ...
    def canonical_id(self, raw: RawDocument) -> str: ...
    def already_ingested(self, document_id: str) -> bool: ...  # checks state.json
```

---

## Supported Sources — Current + Planned

| Source | API / Portal | Status | Format | Notes |
|---|---|---|---|---|
| EU Parliament debates | EP Open Data Portal + EurLex | ✅ existing | JSON | `ep_debates.py` |
| UK Legislation | legislation.gov.uk JSON API | ✅ existing | JSON | Acts + SIs |
| EUR-Lex | REST API (CELEX search) | ✅ existing | JSON/XML | Full EU legislative corpus |
| UK Hansard | hansard.parliament.uk | 🔲 planned (t_d3dc1591) | JSON | Commons + Lords debates |
| Bundestag DIP | dip.bundestag.de | 🔲 planned (t_25a47951) | JSON | German parliamentary debates |
| Assemblée Nationale | assemblee-nationale.fr | 🔲 backlog | XML | French |
| UN Documents | undocs.org / ODS | 🔲 backlog | PDF/XML | Resolutions, reports |
| European Commission | EUR-Lex COM docs | 🔲 backlog | PDF/HTML | Commission proposals |

---

## Reference Resolver

When a document references another document, `reference_resolver.py` extracts the reference and queues it for scraping:

- **CELEX numbers** (EU): `32022R0001` → `eurlex` adapter
- **UK SI references**: `S.I. 2026/458` → `uk_legislation` adapter
- **EP document references**: `A9-0001/2022` → `ep_debates` adapter
- **UN doc symbols**: `A/RES/77/1` → `un_documents` adapter (future)

This enables **document graph expansion**: starting from one debate, the system can recursively retrieve all referenced legislation, building a complete context graph for a policy area.

---

## State Management

Each source maintains a `data/raw/{source}/state.json`:

```json
{
  "last_run": "2026-05-04T10:00:00Z",
  "ingested_ids": ["CRE-9-2022-06-08-ITM-003", ...],
  "failed_ids": [],
  "total_documents": 847
}
```

This enables idempotent re-runs and incremental updates.

---

## Related Tasks

- t_adcd4584 — create services/search/ from scrapers/
- t_25a47951 — Bundestag DIP scraper
- t_d3dc1591 — UK Hansard scraper
- t_56b6a731 — cross-parliament reference following
- t_e842525d — Bundestag raw scraper (predecessor)

## Links

[[product-architecture-v2]] [[service-design-agent]] [[graphgen-architecture]]
