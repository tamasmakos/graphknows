# GraphKnows

Agentic memory for LLM applications: it reads a conversation or document stream,
builds a **knowledge graph** of entities, relations and temporal facts over
[ArcadeDB](https://arcadedb.com/), and serves **hybrid retrieval** (vector ANN +
graph traversal + BM25, fused with RRF) back to your agent.

Two extraction modes:

- **`llm_free`** — deterministic extraction (GLiNER2 entities + spaCy relations) and
  deterministic retrieval. No LLM in the ingest or retrieval path; local embeddings by
  default, so it runs with **zero API keys**.
- **`llm_assisted`** — LLM-driven extraction with an **injectable RDF/OWL ontology**, so
  the graph stays domain-specific. Adds `pip install "graphknows[assisted]"`.

Use it three ways from the same core: as a **Python library**, as an **MCP stdio server**
(Claude Desktop / Claude Code), or as a **web Memory Mirror**.

## Install

```bash
pip install graphknows            # llm_free, batteries included
pip install "graphknows[assisted]"  # + LLM extraction & ontology injection
```

GraphKnows needs a running ArcadeDB. The quickest path:

```bash
docker compose up -d arcadedb
```

> First run downloads the GLiNER2 model and a spaCy model
> (`python -m spacy download en_core_web_lg`). See [docs/configuration.md](docs/configuration.md).

## Quickstart

```python
import asyncio
from graphknows import GraphKnowsRuntime

async def main():
    async with GraphKnowsRuntime() as rt:
        await rt.ingest_memory("Alex joined Acme as CTO in March.", session_id="s1")
        hits = await rt.recall_memory("Where does Alex work?", session_id="s1")
        print(hits)

asyncio.run(main())
```

Configuration is via `GRAPHKNOWS_*` environment variables (or a `.env` file) — see
[.env.example](.env.example) and [docs/configuration.md](docs/configuration.md).

## Running the servers

The same runtime backs an MCP stdio server and a web Memory Mirror.

```bash
docker compose up -d            # arcadedb + mcp + web (Memory Mirror on :8800)
```

**MCP** (Claude Desktop / Claude Code) — register the stdio server:

```json
{
  "mcpServers": {
    "graphknows": {
      "command": "graphknows-mcp",
      "env": { "GRAPHKNOWS_ARCADEDB_URL": "http://localhost:2480" }
    }
  }
}
```

**Web** — `graphknows-web` serves the Memory Mirror (liveness at `/healthz`,
readiness at `/readyz`). The bundled SPA ships via the Docker image; a bare
`pip` install exposes the JSON API and can point at a static dir with
`GRAPHKNOWS_FRONTEND_DIR`.

## Documentation

- [docs/architecture.md](docs/architecture.md) — STM/LTM, promotion, RRF retrieval
- [docs/modes.md](docs/modes.md) — llm_free vs llm_assisted
- [docs/ontology.md](docs/ontology.md) — injecting a domain ontology
- [docs/configuration.md](docs/configuration.md) — full settings reference

The `evaluation/` directory is internal LoCoMo benchmarking — not part of the installed
package.

## License

[Apache-2.0](LICENSE).
