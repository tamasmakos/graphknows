---
title: "Service Design — services/search/"
created: 2026-05-04
updated: 2026-05-20
type: architecture
tags: [architecture, search, connectors, service-design, eu]
confidence: high
---

# Service Design — `services/search/`

The search service is a **document discovery and retrieval service** for official EU and UK public sources. It is not a web crawler — it targets structured official APIs and portals only, returning parsed, chunked `ParsedDocument` objects.

---

## Design Philosophy

- **Official sources only**: parliamentary debates, legislation, regulations, press releases, court decisions, government reports. No news, no social media.
- **Idempotent ingestion**: each document has a stable canonical ID (e.g. CELEX number, EP document reference). Re-fetching the same document is a no-op at the connector level.
- **Async job queue**: scrape jobs are long-running. `POST /search` returns results directly for fast sources; a job-queue model (`POST /jobs`, `GET /jobs/{id}`) handles slow scraping connectors.
- **Connector pattern**: each source is a self-contained class inheriting `SourceConnector` ABC.

---

## Directory Layout

```
services/search/
  main.py                  # FastAPI app factory
  router.py                # Route definitions: /search, /health, /sources
  job_queue.py             # Async job queue (asyncio, in-memory for dev)
  connector_registry.py    # ConnectorRegistry: name → connector instance lookup
  source_manifest.yaml     # Declarative list of all registered sources + metadata
  archive_engine.py        # ArchiveEngine: batch ingestion orchestration
  parsed_document.py       # ParsedDocument dataclass + markdown_to_chunks()
  models.py                # SearchRequest, SearchResult Pydantic models
  mcp_server.py            # MCP server exposing search as a tool

  connectors/
    base.py                # SourceConnector ABC + ScrapingConnector + HttpMixin
    sources/               # One file per source
      ep_debates.py        # EU Parliament plenary debates (EurLex + EP Open Data)
      eurlex.py            # EUR-Lex legislation (CELEX, OJ references)
      ep_resolutions.py    # EP legislative resolutions
      ep_open_data.py      # EP Open Data Portal (general EP documents)
      council_conclusions.py  # Council of the EU conclusions + presidency documents
      ec_press.py          # European Commission press releases (ec.europa.eu/commission/presscorner)
      ecb_publications.py  # ECB publications (Economic Bulletin, Working Papers)
      eca_reports.py       # European Court of Auditors reports
      eprs_briefings.py    # EPRS briefings + in-depth analyses (PDF via CELLAR)
      cellar_sparql.py     # EU CELLAR SPARQL endpoint (metadata + document graph)
      data_europa.py       # data.europa.eu open data portal
      uk_legislation.py    # UK Acts + Statutory Instruments (legislation.gov.uk API)
      wikipedia_eu.py      # Wikipedia articles on EU topics (fallback enrichment)
```

**13 connectors total** as of 2026-05-20.

---

## Connector Hierarchy (connectors/base.py)

```
SourceConnector (ABC)
  ├── search(query, max_results) → list[ParsedDocument]    # mandatory
  └── fetch(doc_id) → ParsedDocument | None                # mandatory

ScrapingConnector(SourceConnector)
  ├── _scrape(query, max_results) → list[ParsedDocument]   # implement this
  └── _fetch_one(doc_id) → ParsedDocument | None           # optionally override

HttpMixin
  └── shared HTTP helpers: rate limiting, retries, HTML→Markdown, PDF→Markdown, lang detect
```

**Disk-cached connectors** (EurLex, EP Debates, UK Legislation) inherit `SourceConnector` directly and implement `search()` + `fetch()` themselves with local file caching.

**Live scraping connectors** (all others) inherit `ScrapingConnector` + `HttpMixin` and implement `_scrape()`.

---

## ParsedDocument

The universal output shape for all connectors:

```python
@dataclass
class ParsedDocument:
    id: str            # canonical source ID (CELEX number, EP ref, etc.)
    title: str
    text: str          # full text as Markdown
    url: str
    source: str        # connector name (e.g. "eprs_briefings")
    date: str | None
    metadata: dict     # source-specific extras (committee, rapporteur, procedure, etc.)
```

`markdown_to_chunks(doc, chunk_size=400, overlap=80)` splits text into overlapping chunks for downstream embedding.

---

## HTTP API

| Method | Path | Description |
|---|---|---|
| `POST` | `/search` | Search across specified sources; returns list of ParsedDocuments |
| `GET` | `/sources` | List all registered connector names + metadata |
| `GET` | `/health` | Health check |
| `POST` | `/jobs` | Queue a long-running batch scrape job |
| `GET` | `/jobs/{job_id}` | Poll job status + results |

---

## Key Notes

- **User-Agent**: all connectors must send `User-Agent: Graphknows/1.0 (https://graphknows.io; research@graphknows.io)`. Wikipedia, CELLAR, and Europa.eu require this. EUR-Lex full text (`eur-lex.europa.eu`) is WAF-blocked and returns empty 202 — use CELLAR SPARQL for EUR-Lex metadata instead.
- **No Bundestag, no UK Hansard yet**: these were planned but are not implemented. Do not reference them as existing.
- **Tavily** is not a connector in this service — it is called directly from `services/agent/tools/search/web.py` as a web fallback.
- Port: `8040` on host (docker-compose default `${SEARCH_PORT:-8040}`). Internal Docker network: `http://search:8000`.
