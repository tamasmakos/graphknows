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
pip install graphknows              # full llm_free stack, MCP server and web UI
pip install "graphknows[assisted]"  # + LLM extraction & injectable ontology
```

| Extra | Adds |
| --- | --- |
| `assisted` | DSPy LLM extraction + injectable RDF/OWL ontology (`llm_assisted` mode) |

`pip install graphknows` is batteries-included: GLiNER2 extraction, local
embeddings, community detection, document parsers, the MCP server and the
Memory Mirror web UI are all base dependencies. `assisted` is the only extra.

The `graphknows.integrations.client` SDK (drive the server out-of-process) needs no extras.

GraphKnows needs a running ArcadeDB. The quickest path:

```bash
docker compose up -d arcadedb
```

> First run downloads the GLiNER2 model and a spaCy model
> (`python -m spacy download en_core_web_lg`). See [docs/configuration.md](docs/configuration.md).

## Quickstart

`Memory` is the single entrypoint — in-process, transport-neutral:

```python
import asyncio
from graphknows import Memory

async def main():
    async with Memory() as mem:
        await mem.ingest_memory("Alex joined Acme as CTO in March.", session_id="s1")
        hits = await mem.recall_memory("Where does Alex work?", session_id="s1")
        print(hits)

asyncio.run(main())
```

Out-of-process (no ML deps in your process), drive the MCP server with the same verbs:

```python
from graphknows.integrations.client import RemoteMemory

async with RemoteMemory(command="graphknows-mcp") as mem:
    await mem.ingest_memory("Alex joined Acme as CTO in March.", session_id="s1")
    hits = await mem.recall_memory("Where does Alex work?", session_id="s1")
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

## Integrating into an agent framework

Wire your framework's memory hooks to the `Memory`/`RemoteMemory` verbs. The
LangGraph adapter is shipped as the reference; the generic recipe (write →
`ingest_memory`, read → `recall_memory`, thread id → `session_id`, end of
episode → `flush_memory`) covers any framework — see
[docs/integrations.md](docs/integrations.md) and
[examples/langgraph_agent.py](examples/langgraph_agent.py).

To plug a different storage backend, implement `IGraphStore` from
[graphknows/storage/base.py](graphknows/storage/base.py) — the single-database
contract `Memory` actually uses.

## Documentation

- [docs/architecture.md](docs/architecture.md) — STM/LTM, promotion, RRF retrieval
- [docs/modes.md](docs/modes.md) — llm_free vs llm_assisted
- [docs/ontology.md](docs/ontology.md) — injecting a domain ontology
- [docs/configuration.md](docs/configuration.md) — full settings reference
- [docs/integrations.md](docs/integrations.md) — framework integration recipe
- [docs/versioning.md](docs/versioning.md) — semver & deprecation policy

The `evaluation/` directory is internal LoCoMo benchmarking — not part of the installed
package.

## License

[Apache-2.0](LICENSE).
